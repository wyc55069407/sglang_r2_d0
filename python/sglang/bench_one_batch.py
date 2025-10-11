"""
Benchmark the latency of running a single static batch without a server.

This script does not launch a server and uses the low-level APIs.
It accepts server arguments (the same as launch_server.py) and benchmark arguments (e.g., batch size, input lengths).

# Usage (latency test)
## with dummy weights:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --load-format dummy
## sweep through multiple data points and store (append) the results in a jsonl file:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 12 14 --input-len 256 512 --output-len 32 256 --run-name test_run
## run with profiling:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 12 14 --input-len 256 512 --profile
# Usage (correctness test):
python -m sglang.bench_one_batch --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4 --correct

## Reference output (of the correctness test above, can be gpu dependent):
input_ids=[[1, 450, 7483, 310, 3444, 338], [1, 450, 7483, 310, 278, 3303, 13187, 290, 338], [1, 20628, 338, 263, 6575, 1460, 2462, 322, 306, 763]]

prefill logits (first half): tensor([[-10.0312,  -9.5000,   0.8931,  ...,  -4.9414,  -3.2422,  -3.3633],
        [-10.0312,  -9.5000,   0.8931,  ...,  -4.9414,  -3.2422,  -3.3633],
        [ -9.1875, -10.2500,   2.7129,  ...,  -4.3359,  -4.0664,  -4.1328]],
       device='cuda:0')

prefill logits (final): tensor([[-8.3125, -7.1172,  3.3457,  ..., -4.9570, -4.1328, -3.4141],
        [-8.9141, -9.0156,  4.1445,  ..., -4.9922, -4.4961, -4.0781],
        [-9.6328, -9.0547,  4.0195,  ..., -5.3047, -4.7148, -4.4570]],
       device='cuda:0')

========== Prompt 0 ==========
<s> The capital of France is Paris.
The capital of the United States is Washington, D.C.


========== Prompt 1 ==========
<s> The capital of the United Kindom is London.
The capital of the United Kingdom is London.
The capital of the

========== Prompt 2 ==========
<s> Today is a sunny day and I like to go for a walk in the park.
I'm going to the park
"""

import argparse
import dataclasses
import itertools
import json
import logging
import multiprocessing
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import destroy_distributed_environment
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    configure_logger,
    get_bool_env_var,
    kill_process_tree,
    set_gpu_proc_affinity,
    suppress_other_loggers,
)


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    result_filename: str = "result.jsonl"
    correctness_test: bool = False
    # This is only used for correctness test
    cut_len: int = 4
    log_decode_step: int = 0
    profile: bool = False
    profile_filename_prefix: str = "profile"
    iterations: int = 1

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=BenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument("--correctness-test", action="store_true")
        parser.add_argument("--cut-len", type=int, default=BenchArgs.cut_len)
        parser.add_argument(
            "--log-decode-step",
            type=int,
            default=BenchArgs.log_decode_step,
            help="Log decode latency by step, default is set to zero to disable.",
        )
        parser.add_argument(
            "--profile", action="store_true", help="Use Torch Profiler."
        )
        parser.add_argument(
            "--profile-filename-prefix",
            type=str,
            default=BenchArgs.profile_filename_prefix,
            help="Prefix of the profiling file names. The full profiling result file(s) be "
            '"[profile_filename_prefix]_batch[batch_size]_input[input_len]_output[output_len].trace.json.gz"',
        )
        parser.add_argument(
            "--iterations",
            type=int,
            default=BenchArgs.iterations,
            help="Number of iterations to run in the latency benchmark. Default is 1.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


def load_model(server_args, port_args, tp_rank):
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    model_config = ModelConfig.from_server_args(server_args)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    rank_print(f"max_total_num_tokens={model_runner.max_total_num_tokens}")
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    if server_args.tp_size > 1:
        dist.barrier()
    return model_runner, tokenizer


def prepare_inputs_for_correctness_test(bench_args, tokenizer, batch_size):
    prompts = [
        """
        Outstanding European Travel Plans

The Classic European Capitals Adventure: A 14-Day Odyssey of History, Art, and Culture

Europe, a continent steeped in millennia of history, brimming with artistic expression, and pulsating with diverse cultures, has captivated travelers for centuries. For the first-time European explorer, the sheer volume of potential destinations can be overwhelming. Where to begin? This meticulously crafted 14-day itinerary provides a perfect introduction to the continent's most iconic capital cities: the romantic allure of Paris, the regal grandeur of London, and the ancient majesty of Rome.

This is not merely a superficial checklist of famous landmarks; it's a carefully orchestrated odyssey designed to ignite the senses and leave an enduring impression. It's a journey that delves beneath the surface, exploring hidden neighborhoods, engaging with local artisans, and savoring authentic culinary experiences. The pace is dynamic, designed to maximize your time, but the rewards are unparalleled. Imagine yourself transported back in time within the opulent Palace of Versailles, feeling the weight of history within the formidable Tower of London, and standing in awe before the ancient grandeur of the Roman Forum. All of this, and much more, is achievable within a captivating fortnight. This itinerary is crafted not just to see, but to truly experience Europe.

Duration: 14 Days / 13 Nights

Theme: History, Art, Culture, Iconic Landmarks, Culinary Delights, Hidden Gems, Local Experiences

Destinations: Paris (France), London (England), Rome (Italy)

Plan at a Glance:

Days 1-3: Paris, France – Beyond the Postcard: Unveiling the Parisian Soul. We'll delve deeper than the iconic landmarks, exploring hidden neighborhoods, engaging with local artisans, indulging in authentic culinary experiences, and truly immersing ourselves in the City of Lights' unique charm.

Days 4-6: London, England – From Royal Grandeur to Modern Vibrancy: Discovering London's Multifaceted Identity. This segment goes beyond the royal sights, exploring historical depths, embracing the vibrant cultural scene, experiencing the city's diverse culinary offerings, and uncovering its hidden pockets of creativity.

Days 7-9: Rome, Italy – Walking Through Time: Experiencing the Eternal City's Ancient Majesty and Artistic Renaissance. We'll step back through millennia to witness the awe-inspiring marvels of the Roman Empire, connect with the spiritual heart of the Catholic Church, admire Renaissance masterpieces, and savor the authentic flavors of Roman cuisine.

Days 10-11: Travel Days/Flex Days – Unscripted Moments: Customizing Your European Adventure. These days offer the flexibility to adapt to your individual pace and interests, allowing for comfortable travel, well-deserved rest, or the opportunity to delve deeper into the cities that resonate most profoundly.

Days 12-14: Departure – Reflecting on the Journey: Taking Europe Home. As you prepare for your return, we'll encourage reflection on the transformative experiences, unforgettable moments, and lasting memories created throughout this European adventure.

Detailed Itinerary:

Day 1-3: Paris, France - The City of Lights: A Parisian Rhapsody – Extended Edition

Paris, the City of Lights, is a city of dreams, romance, and unparalleled beauty. But to truly understand Paris, one must venture beyond the postcard-perfect images. It's in the hidden courtyards, the bustling markets, and the charming bistros that the true Parisian soul resides. This extended itinerary is designed to capture that essence, inviting you to immerse yourself in the city's unique charm and discover its hidden gems.

Accommodation:

Choosing the right accommodation is crucial for a fulfilling Parisian experience. Different neighborhoods offer distinct atmospheres and price points.

Marais District: History, Elegance, and Hidden Delights: Located in the heart of Paris, the Marais is a historical treasure trove. It's a neighborhood of elegant mansions ("hôtels particuliers"), hidden courtyards, and a rich Jewish heritage. Beyond the trendy boutiques and art galleries that now populate its streets, the Marais whispers tales of aristocracy, revolution, and resilience.

Place des Vosges: Begin your exploration at the Place des Vosges, one of the most beautiful squares in Paris. This meticulously planned square, with its harmonious architecture and central park, provides a serene oasis in the bustling city. Constructed in the early 17th century, it was once a popular spot for aristocratic duels and festivities. Today, it's a place to relax, admire the architecture, and soak up the Parisian atmosphere.

Musée Carnavalet: Delve deeper into the city's history at the Musée Carnavalet, dedicated to the history of Paris. Housed in two magnificent hôtels particuliers, the museum showcases artifacts, paintings, and documents that tell the story of Paris from its earliest settlements to the present day. This is the perfect place to gain a deeper understanding of the city's evolution and its cultural identity.

Jewish Quarter (Pletzl): Explore the historic Jewish Quarter, known as the Pletzl. This area has been a center of Jewish life in Paris for centuries, and it's filled with synagogues, kosher restaurants, and shops selling traditional Jewish goods. Wander through the narrow streets and experience the vibrant culture of this unique community.

Accommodation Recommendations: Consider staying in a boutique hotel housed in a restored 17th-century building in the Marais. Many of these hotels offer charming rooms with historical details and a sense of Parisian elegance. Look for hotels with courtyards or gardens for a peaceful retreat from the city's hustle and bustle.

Latin Quarter: Student Life, Literary History, and Bohemian Charm: On the Left Bank of the Seine, the Latin Quarter pulses with intellectual energy and bohemian spirit. Historically the home of the Sorbonne University and numerous prestigious schools, it's a neighborhood that has nurtured generations of thinkers, writers, and artists. The Latin Quarter is more than just a student hangout; it's a place steeped in literary history and artistic expression.

Shakespeare and Company: No literary pilgrimage to Paris is complete without a visit to Shakespeare and Company, the iconic English-language bookstore. This legendary bookstore has been a haven for writers and intellectuals for decades, and it continues to be a vibrant hub for literary culture. Browse the shelves, attend a reading, or simply soak up the atmosphere of this literary landmark.

Sorbonne University: Explore the Sorbonne University, one of the oldest and most prestigious universities in Europe. While access to the interior may be limited, admire the architecture of the buildings and soak up the intellectual atmosphere of the campus.

Panthéon: Visit the Panthéon, a neoclassical monument that houses the tombs of famous French figures, including Voltaire, Rousseau, Victor Hugo, and Marie Curie. This impressive building is a testament to French intellectual and cultural achievements.

Luxembourg Gardens: Wander through the Luxembourg Gardens, a beautiful park that offers a respite from the city's hustle and bustle. Admire the formal gardens, relax by the fountains, or take a stroll along the tree-lined paths.

Accommodation Recommendations: Consider staying in a historic hotel in the Latin Quarter with a literary past. Many hotels in this neighborhood have been frequented by writers and intellectuals for centuries, and they offer a unique and atmospheric experience.

Considerations for Accommodation:

Research: Thoroughly research accommodation options based on your preferred atmosphere, budget, and accessibility requirements. Read reviews from other travelers to get a sense of the quality and service of different hotels and apartments.

Book in Advance: Book your accommodation well in advance, especially if you're traveling during peak season (summer, holidays, fashion week). Paris is a popular destination, and the best hotels and apartments tend to fill up quickly.

Location and Transportation: Consider the proximity of your accommodation to metro stations, bus stops, and other transportation options. Paris has an excellent public transportation system, but being close to a metro station will make it easier to get around the city. Also, factor in the walkability of the neighborhood; some neighborhoods are more pedestrian-friendly than others.

Amenities: Look for hotels with amenities that are important to you, such as air conditioning (especially during the summer months), free Wi-Fi, breakfast included, and a concierge service.

Personal Touch: Consider staying in a smaller, family-run hotel for a more personal and authentic experience. These hotels often offer a more intimate atmosphere and a higher level of personalized service.

Activities:

Paris offers an endless array of activities, from iconic landmarks to hidden gems. This itinerary provides a framework for your exploration, but feel free to customize it to your interests and preferences.

Arrival & Seine Stroll: Beyond the Bridges: After checking into your hotel, begin your Parisian adventure with a leisurely stroll along the Seine River.

Guided Walking Tour: Instead of simply walking along the riverbank, consider taking a guided walking tour that focuses on the history and architecture of the Seine's bridges. These tours provide fascinating insights into the construction, significance, and artistic details of these iconic structures.

Stories Behind the Bridges: Learn about the history of each bridge, its unique architectural style, and the stories behind the statues and sculptures that adorn them. Discover the Pont Neuf, the oldest bridge in Paris, and the Pont Alexandre III, one of the most elegant.

Different Perspectives: A guided tour will provide a deeper appreciation for the Seine and its role in Parisian history and culture.

Eiffel Tower: Evening Illumination and Hidden History: No trip to Paris is complete without a visit to the Eiffel Tower.

Pre-Booking: Pre-booking tickets is essential to avoid long queues, especially during peak season. Book your tickets online in advance to secure your preferred time slot.

Off-Peak Hours: Consider visiting the Eiffel Tower during off-peak hours, such as early morning or late evening, to avoid the biggest crowds.

Construction and Role: Learn about the tower's construction, its role in the 1889 World's Fair, and its evolution as a symbol of Paris.

Taking the Stairs: For a more challenging but rewarding experience, consider taking the stairs instead of the elevator to the first or second level.

Picnic on the Champ de Mars: Enjoy a picnic on the Champ de Mars with the Eiffel Tower as your backdrop. This is a classic Parisian experience that allows you to relax and soak up the atmosphere.

Guided Tours: Consider a guided tour that focuses on the history and engineering of the Eiffel Tower. These tours provide fascinating details about the tower's design, construction, and cultural significance.

Louvre Museum: Focusing Your Visit and Discovering Hidden Gems: The Louvre Museum is one of the largest and most famous museums in the world, housing an unparalleled collection of art from around the globe.

Strategic Planning: It's impossible to see everything in the Louvre in one day. Plan your visit carefully and focus on the areas that interest you most. Download a map of the museum and identify the galleries you want to visit.

Guided Tours and Masterpieces: Consider a guided tour that highlights the museum's masterpieces and hidden gems. A knowledgeable guide can help you navigate the vast collection and provide insights into the art and history.

Mobile App: Download the Louvre's mobile app for interactive maps, audio guides, and information about the museum's collections.

Less Crowded Wings: Explore the museum's less crowded wings, such as the Egyptian antiquities or the Islamic art collections. These areas often offer a more intimate and rewarding experience.

Tuileries Garden: Visit the Tuileries Garden, located next to the Louvre, for a relaxing stroll. This beautiful garden provides a tranquil escape from the crowds of the museum.

Notre Dame Cathedral: Reflections and Remembrance: Even while under reconstruction, Notre Dame Cathedral remains a powerful symbol of Paris.

Memorial Site: Visit the memorial site and reflect on the cathedral's history and its significance to the city. Take a moment to appreciate the architectural beauty of the cathedral's exterior, even as it undergoes restoration.

Île de la Cité: Walk around the Île de la Cité, the island on which Notre Dame is located, and explore the surrounding area, including the Conciergerie, a former royal palace and prison.

Nearby Churches: Consider attending a service at a nearby church, such as Saint-Germain-des-Prés or Saint-Séverin, to experience the spiritual side of Paris.

Sainte-Chapelle: A Kaleidoscope of Light and History: Sainte-Chapelle is a masterpiece of Gothic architecture, renowned for its stunning stained-glass windows.

Ample Time: Allocate ample time to admire the intricate details of the stained-glass windows and learn about their biblical themes. The windows depict scenes from the Old and New Testaments, and they are a testament to the skill and artistry of the medieval craftsmen who created them.

Conciergerie: Visit the Conciergerie, located next door, to learn about its history as a royal palace and prison. The Conciergerie was once the home of French kings, and it later served as a prison during the French Revolution.

Concerts: Consider attending a concert at Sainte-Chapelle for a truly unforgettable experience. The acoustics in the chapel are superb, and the setting is magical.

Montmartre & Sacré-Cœur Basilica: Artistic Inspiration and Panoramic Views: Montmartre, the highest point in Paris, is a neighborhood known for its artistic history and its stunning views of the city.

Walking Tour: Take a walking tour of Montmartre and discover its hidden streets, artists' studios, and charming cafes.

Musée de Montmartre: Visit the Musée de Montmartre to learn about the history of the neighborhood and its artistic heritage. The museum is housed in a former artists' residence, and it showcases paintings, drawings, and photographs that depict life in Montmartre.

Sacré-Cœur Basilica: Climb to the Sacré-Cœur Basilica for panoramic views of the city. The basilica is a stunning example of Romanesque-Byzantine architecture, and it's one of the most iconic landmarks in Paris.

Crepes: Enjoy a crepe from a street vendor. Montmartre is famous for its crepes, and they're the perfect snack to enjoy while exploring the neighborhood.

Attend a Service: Consider attending a service at the Sacré-Cœur Basilica.

Palace of Versailles: Beyond the Hall of Mirrors: A day trip to Versailles, the former royal palace, is an essential part of any Parisian experience.

Lavish Interiors: Explore the palace's lavish interiors, including the Hall of Mirrors, the Royal Apartments, and the Chapel. The Hall of Mirrors is one of the most famous rooms in the palace, and it's a testament to the opulence and grandeur of the French monarchy.

Vast Gardens: Wander through the vast and meticulously manicured gardens, including the Grand Trianon and the Petit Trianon.

Bike Rental: Rent a bike to explore the gardens more efficiently. The gardens are vast, and a bike is a great way to see more of them.

Queen's Hamlet: Visit the Queen's Hamlet, a picturesque village built for Marie Antoinette. This charming village provides a glimpse into the Queen's private life.

Picnic Lunch: Pack a picnic lunch to enjoy amidst the grandeur. There are several picnic areas in the gardens where you can relax and enjoy the scenery.

Guided Tour: Consider a guided tour that focuses on the history and architecture of Versailles.

Seine River Cruise: Romantic Views and Parisian Charm: Conclude your Parisian adventure with a Seine River cruise.

Commentary: Choose a cruise that offers commentary on the landmarks you pass.

Dinner Cruise: Enjoy a dinner cruise for a romantic experience.

Live Music: Listen to live music on board.

Summary:
""",
"""Harry Potter and the Sorcerer\'s Stone


CHAPTER ONE

THE BOY WHO LIVED

Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say
that they were perfectly normal, thank you very much. They were the last
people you\'d expect to be involved in anything strange or mysterious,
because they just didn\'t hold with such nonsense.

Mr. Dursley was the director of a firm called Grunnings, which made
drills. He was a big, beefy man with hardly any neck, although he did
have a very large mustache. Mrs. Dursley was thin and blonde and had
nearly twice the usual amount of neck, which came in very useful as she
spent so much of her time craning over garden fences, spying on the
neighbors. The Dursleys had a small son called Dudley and in their
opinion there was no finer boy anywhere.

The Dursleys had everything they wanted, but they also had a secret, and
their greatest fear was that somebody would discover it. They didn\'t
think they could bear it if anyone found out about the Potters. Mrs.
Potter was Mrs. Dursley\'s sister, but they hadn\'t met for several years;
in fact, Mrs. Dursley pretended she didn\'t have a sister, because her
sister and her good-for-nothing husband were as unDursleyish as it was
possible to be. The Dursleys shuddered to think what the neighbors would
say if the Potters arrived in the street. The Dursleys knew that the
Potters had a small son, too, but they had never even seen him. This boy
was another good reason for keeping the Potters away; they didn\'t want
Dudley mixing with a child like that.

When Mr. and Mrs. Dursley woke up on the dull, gray Tuesday our story
starts, there was nothing about the cloudy sky outside to suggest that
strange and mysterious things would soon be happening all over the
country. Mr. Dursley hummed as he picked out his most boring tie for
work, and Mrs. Dursley gossiped away happily as she wrestled a screaming
Dudley into his high chair.

None of them noticed a large, tawny owl flutter past the window.

At half past eight, Mr. Dursley picked up his briefcase, pecked Mrs.
Dursley on the cheek, and tried to kiss Dudley good-bye but missed,
because Dudley was now having a tantrum and throwing his cereal at the
walls. "Little tyke," chortled Mr. Dursley as he left the house. He got
into his car and backed out of number four\'s drive.

It was on the corner of the street that he noticed the first sign of
something peculiar -- a cat reading a map. For a second, Mr. Dursley
didn\'t realize what he had seen -- then he jerked his head around to
look again. There was a tabby cat standing on the corner of Privet
Drive, but there wasn\'t a map in sight. What could he have been thinking
of? It must have been a trick of the light. Mr. Dursley blinked and
stared at the cat. It stared back. As Mr. Dursley drove around the
corner and up the road, he watched the cat in his mirror. It was now
reading the sign that said Privet Drive -- no, looking at the sign; cats
couldn\'t read maps or signs. Mr. Dursley gave himself a little shake and
put the cat out of his mind. As he drove toward town he thought of
nothing except a large order of drills he was hoping to get that day.

But on the edge of town, drills were driven out of his mind by something
else. As he sat in the usual morning traffic jam, he couldn\'t help
noticing that there seemed to be a lot of strangely dressed people
about. People in cloaks. Mr. Dursley couldn\'t bear people who dressed in
funny clothes -- the getups you saw on young people! He supposed this
was some stupid new fashion. He drummed his fingers on the steering
wheel and his eyes fell on a huddle of these weirdos standing quite
close by. They were whispering excitedly together. Mr. Dursley was
enraged to see that a couple of them weren\'t young at all; why, that man
had to be older than he was, and wearing an emerald-green cloak! The
nerve of him! But then it struck Mr. Dursley that this was probably some
silly stunt -- these people were obviously collecting for something...
yes, that would be it. The traffic moved on and a few minutes later, Mr.
Dursley arrived in the Grunnings parking lot, his mind back on drills.

Mr. Dursley always sat with his back to the window in his office on the
ninth floor. If he hadn\'t, he might have found it harder to concentrate
on drills that morning. He didn\'t see the owls swoop ing past in broad
daylight, though people down in the street did; they pointed and gazed
open- mouthed as owl after owl sped overhead. Most of them had never
seen an owl even at nighttime. Mr. Dursley, however, had a perfectly
normal, owl-free morning. He yelled at five different people. He made
several important telephone calls and shouted a bit more. He was in a
very good mood until lunchtime, when he thought he\'d stretch his legs
and walk across the road to buy himself a bun from the bakery.

He\'d forgotten all about the people in cloaks until he passed a group of
them next to the baker\'s. He eyed them angrily as he passed. He didn\'t
know why, but they made him uneasy. This bunch were whispering
excitedly, too, and he couldn\'t see a single collecting tin. It was on
his way back past them, clutching a large doughnut in a bag, that he
caught a few words of what they were saying.

"The Potters, that\'s right, that\'s what I heard yes, their son, Harry"

Mr. Dursley stopped dead. Fear flooded him. He looked back at the
whisperers as if he wanted to say something to them, but thought better
of it.

He dashed back across the road, hurried up to his office, snapped at his
secretary not to disturb him, seized his telephone, and had almost
finished dialing his home number when he changed his mind. He put the
receiver back down and stroked his mustache, thinking... no, he was
being stupid. Potter wasn\'t such an unusual name. He was sure there were
lots of people called Potter who had a son called Harry. Come to think
of it, he wasn\'t even sure his nephew was called Harry. He\'d never even
seen the boy. It might have been Harvey. Or Harold. There was no point
in worrying Mrs. Dursley; she always got so upset at any mention of her
sister. He didn\'t blame her -- if he\'d had a sister like that... but all
the same, those people in cloaks...

He found it a lot harder to concentrate on drills that afternoon and
when he left the building at five o\'clock, he was still so worried that
he walked straight into someone just outside the door.

"Sorry," he grunted, as the tiny old man stumbled and almost fell. It
was a few seconds before Mr. Dursley realized that the man was wearing a
violet cloak. He didn\'t seem at all upset at being almost knocked to the
ground. On the contrary, his face split into a wide smile and he said in
a squeaky voice that made passersby stare, "Don\'t be sorry, my dear sir,
for nothing could upset me today! Rejoice, for You-Know-Who has gone at
last! Even Muggles like yourself should be celebrating, this happy,
happy day!"

And the old man hugged Mr. Dursley around the middle and walked off.

Mr. Dursley stood rooted to the spot. He had been hugged by a complete
stranger. He also thought he had been called a Muggle, whatever that
was. He was rattled. He hurried to his car and set off for home, hoping
he was imagining things, which he had never hoped before, because he
didn\'t approve of imagination.

As he pulled into the driveway of number four, the first thing he saw --
and it didn\'t improve his mood -- was the tabby cat he\'d spotted that
morning. It was now sitting on his garden wall. He was sure it was the
same one; it had the same markings around its eyes.

"Shoo!" said Mr. Dursley loudly. The cat didn\'t move. It just gave him a
stern look. Was this normal cat behavior? Mr. Dursley wondered. Trying
to pull himself together, he let himself into the house. He was still
determined not to mention anything to his wife.

Mrs. Dursley had had a nice, normal day. She told him over dinner all
about Mrs. Next Door\'s problems with her daughter and how Dudley had
learned a new word ("Won\'t!"). Mr. Dursley tried to act normally. When
Dudley had been put to bed, he went into the living room in time to
catch the last report on the evening news:

"And finally, bird-watchers everywhere have reported that the nation\'s
owls have been behaving very unusually today. Although owls normally
hunt at night and are hardly ever seen in daylight, there have been
hundreds of sightings of these birds flying in every direction since
sunrise. Experts are unable to explain why the owls have suddenly
changed their sleeping pattern." The newscaster allowed himself a grin.
"Most mysterious. And now, over to Jim McGuffin with the weather. Going
to be any more showers of owls tonight, Jim?"

"Well, Ted," said the weatherman, "I don\'t know about that, but it\'s not
only the owls that have been acting oddly today. Viewers as far apart as
Kent, Yorkshire, and Dundee have been phoning in to tell me that instead
of the rain I promised yesterday, they\'ve had a downpour of shooting
stars! Perhaps people have been celebrating Bonfire Night early -- it\'s
not until next week, folks! But I can promise a wet night tonight."

Mr. Dursley sat frozen in his armchair. Shooting stars all over Britain?
Owls flying by daylight? Mysterious people in cloaks all over the place?
And a whisper, a whisper about the Potters...

Mrs. Dursley came into the living room carrying two cups of tea. It was
no good. He\'d have to say something to her. He cleared his throat
nervously. "Er -- Petunia, dear -- you haven\'t heard from your sister
lately, have you?"

As he had expected, Mrs. Dursley looked shocked and angry. After all,
they normally pretended she didn\'t have a sister.

"No," she said sharply. "Why?"

"Funny stuff on the news," Mr. Dursley mumbled. "Owls... shooting
stars... and there were a lot of funny-looking people in town today..."

"So?" snapped Mrs. Dursley.

"Well, I just thought... maybe... it was something to do with... you
know... her crowd."

Mrs. Dursley sipped her tea through pursed lips. Mr. Dursley wondered
whether he dared tell her he\'d heard the name "Potter." He decided he
didn\'t dare. Instead he said, as casually as he could, "Their son --
he\'d be about Dudley\'s age now, wouldn\'t he?"

"I suppose so," said Mrs. Dursley stiffly.

"What\'s his name again? Howard, isn\'t it?"

"Harry. Nasty, common name, if you ask me."

"Oh, yes," said Mr. Dursley, his heart sinking horribly. "Yes, I quite
agree."

He didn\'t say another word on the subject as they went upstairs to bed.
While Mrs. Dursley was in the bathroom, Mr. Dursley crept to the bedroom
window and peered down into the front garden. The cat was still there.
It was staring down Privet Drive as though it were waiting for
something.

Was he imagining things? Could all this have anything to do with the
Potters? If it did... if it got out that they were related to a pair of
-- well, he didn\'t think he could bear it.

The Dursleys got into bed. Mrs. Dursley fell asleep quickly but Mr.
Dursley lay awake, turning it all over in his mind. His last, comforting
thought before he fell asleep was that even if the Potters were
involved, there was no reason for them to come near him and Mrs.
Dursley. The Potters knew very well what he and Petunia thought about
them and their kind.... He couldn\'t see how he and Petunia could get
mixed up in anything that might be going on -- he yawned and turned over
-- it couldn\'t affect them....

How very wrong he was.

Mr. Dursley might have been drifting into an uneasy sleep, but the cat
on the wall outside was showing no sign of sleepiness. It was sitting as
still as a statue, its eyes fixed unblinkingly on the far corner of
Privet Drive. It didn\'t so much as quiver when a car door slammed on the
next street, nor when two owls swooped overhead. In fact, it was nearly
midnight before the cat moved at all.

A man appeared on the corner the cat had been watching, appeared so
suddenly and silently you\'d have thought he\'d just popped out of the
ground. The cat\'s tail twitched and its eyes narrowed.

Nothing like this man had ever been seen on Privet Drive. He was tall,
thin, and very old, judging by the silver of his hair and beard, which
were both long enough to tuck into his belt. He was wearing long robes,
a purple cloak that swept the ground, and high-heeled, buckled boots.
His blue eyes were light, bright, and sparkling behind half-moon
spectacles and his nose was very long and crooked, as though it had been
broken at least twice. This man\'s name was Albus Dumbledore.

Albus Dumbledore didn\'t seem to realize that he had just arrived in a
street where everything from his name to his boots was unwelcome. He was
busy rummaging in his cloak, looking for something. But he did seem to
realize he was being watched, because he looked up suddenly at the cat,
which was still staring at him from the other end of the street. For
some reason, the sight of the cat seemed to amuse him. He chuckled and
muttered, "I should have known."

He found what he was looking for in his inside pocket. It seemed to be a
silver cigarette lighter. He flicked it open, held it up in the air, and
clicked it. The nearest street lamp went out with a little pop. He
clicked it again -- the next lamp flickered into darkness. Twelve times
he clicked the Put-Outer, until the only lights left on the whole street
were two tiny pinpricks in the distance, which were the eyes of the cat
watching him. If anyone looked out of their window now, even beady-eyed
Mrs. Dursley, they wouldn\'t be able to see anything that was happening
down on the pavement. Dumbledore slipped the Put-Outer back inside his
cloak and set off down the street toward number four, where he sat down
on the wall next to the cat. He didn\'t look at it, but after a moment he
spoke to it.

"Fancy seeing you here, Professor McGonagall."

He turned to smile at the tabby, but it had gone. Instead he was smiling
at a rather severe-looking woman who was wearing square glasses exactly
the shape of the markings the cat had had around its eyes. She, too, was
wearing a cloak, an emerald one. Her black hair was drawn into a tight
bun. She looked distinctly ruffled.

"How did you know it was me?" she asked.

"My dear Professor, I \'ve never seen a cat sit so stiffly."

"You\'d be stiff if you\'d been sitting on a brick wall all day," said
Professor McGonagall.

"All day? When you could have been celebrating? I must have passed a
dozen feasts and parties on my way here."

Professor McGonagall sniffed angrily.

"Oh yes, everyone\'s celebrating, all right," she said impatiently.
"You\'d think they\'d be a bit more careful, but no -- even the Muggles
have noticed something\'s going on. It was on their news." She jerked her
head back at the Dursleys\' dark living-room window. "I heard it. Flocks
of owls... shooting stars.... Well, they\'re not completely stupid. They
were bound to notice something. Shooting stars down in Kent -- I\'ll bet
that was Dedalus Diggle. He never had much sense."

"You can\'t blame them," said Dumbledore gently. "We\'ve had precious
little to celebrate for eleven years."

"I know that," said Professor McGonagall irritably. "But that\'s no
reason to lose our heads. People are being downright careless, out on
the streets in broad daylight, not even dressed in Muggle clothes,
swapping rumors."

She threw a sharp, sideways glance at Dumbledore here, as though hoping
he was going to tell her something, but he didn\'t, so she went on. "A
fine thing it would be if, on the very day YouKnow-Who seems to have
disappeared at last, the Muggles found out about us all. I suppose he
really has gone, Dumbledore?"

"It certainly seems so," said Dumbledore. "We have much to be thankful
for. Would you care for a lemon drop?"

"A what?"

"A lemon drop. They\'re a kind of Muggle sweet I\'m rather fond of"

"No, thank you," said Professor McGonagall coldly, as though she didn\'t
think this was the moment for lemon drops. "As I say, even if
You-Know-Who has gone -"

"My dear Professor, surely a sensible person like yourself can call him
by his name? All this \'You- Know-Who\' nonsense -- for eleven years I
have been trying to persuade people to call him by his proper name:
Voldemort." Professor McGonagall flinched, but Dumbledore, who was
unsticking two lemon drops, seemed not to notice. "It all gets so
confusing if we keep saying \'You-Know-Who.\' I have never seen any reason
to be frightened of saying Voldemort\'s name.

"I know you haven \'t, said Professor McGonagall, sounding half
exasperated, half admiring. "But you\'re different. Everyone knows you\'re
the only one You-Know- oh, all right, Voldemort, was frightened of."

"You flatter me," said Dumbledore calmly. "Voldemort had powers I will
never have."

"Only because you\'re too -- well -- noble to use them."

"It\'s lucky it\'s dark. I haven\'t blushed so much since Madam Pomfrey
told me she liked my new earmuffs."

Professor McGonagall shot a sharp look at Dumbledore and said, "The owls
are nothing next to the rumors that are flying around. You know what
everyone\'s saying? About why he\'s disappeared? About what finally
stopped him?"

It seemed that Professor McGonagall had reached the point she was most
anxious to discuss, the real reason she had been waiting on a cold, hard
wall all day, for neither as a cat nor as a woman had she fixed
Dumbledore with such a piercing stare as she did now. It was plain that
whatever "everyone" was saying, she was not going to believe it until
Dumbledore told her it was true. Dumbledore, however, was choosing
another lemon drop and did not answer.

"What they\'re saying," she pressed on, "is that last night Voldemort
turned up in Godric\'s Hollow. He went to find the Potters. The rumor is
that Lily and James Potter are -- are -- that they\'re -- dead. "

Dumbledore bowed his head. Professor McGonagall gasped.

"Lily and James... I can\'t believe it... I didn\'t want to believe it...
Oh, Albus..."

Dumbledore reached out and patted her on the shoulder. "I know... I
know..." he said heavily.

Professor McGonagall\'s voice trembled as she went on. "That\'s not all.
They\'re saying he tried to kill the Potter\'s son, Harry. But -- he
couldn\'t. He couldn\'t kill that little boy. No one knows why, or how,
but they\'re saying that when he couldn\'t kill Harry Potter, Voldemort\'s
power somehow broke -- and that\'s why he\'s gone.

Dumbledore nodded glumly.

"It\'s -- it\'s true?" faltered Professor McGonagall. "After all he\'s
done... all the people he\'s killed... he couldn\'t kill a little boy?
It\'s just astounding... of all the things to stop him... but how in the
name of heaven did Harry survive?"

"We can only guess," said Dumbledore. "We may never know."

Professor McGonagall pulled out a lace handkerchief and dabbed at her
eyes beneath her spectacles. Dumbledore gave a great sniff as he took a
golden watch from his pocket and examined it. It was a very odd watch.
It had twelve hands but no numbers; instead, little planets were moving
around the edge. It must have made sense to Dumbledore, though, because
he put it back in his pocket and said, "Hagrid\'s late. I suppose it was
he who told you I\'d be here, by the way?"

"Yes," said Professor McGonagall. "And I don\'t suppose you\'re going to
tell me why you\'re here, of all places?"

"I\'ve come to bring Harry to his aunt and uncle. They\'re the only family
he has left now."

"You don\'t mean -- you can\'t mean the people who live here?" cried
Professor McGonagall, jumping to her feet and pointing at number four.
"Dumbledore -- you can\'t. I\'ve been watching them all day. You couldn\'t
find two people who are less like us. And they\'ve got this son -- I saw
him kicking his mother all the way up the street, screaming for sweets.
Harry Potter come and live here!"

"It\'s the best place for him," said Dumbledore firmly. "His aunt and
uncle will be able to explain everything to him when he\'s older. I\'ve
written them a letter."

"A letter?" repeated Professor McGonagall faintly, sitting back down on
the wall. "Really, Dumbledore, you think you can explain all this in a
letter? These people will never understand him! He\'ll be famous -- a
legend -- I wouldn\'t be surprised if today was known as Harry Potter day
in the future -- there will be books written about Harry -- every child
in our world will know his name!"

"Exactly," said Dumbledore, looking very seriously over the top of his
half-moon glasses. "It would be enough to 
""",
        "The capital of France is",
        "The capital of the United Kindom is",
        "Today is a sunny day and I like",
        "Sky is blue because",
        """The Qwen3 Embedding model series is the latest proprietary model of the Qwen family, specifically designed for text embedding and ranking tasks. Building upon the dense foundational models of the Qwen3 series, it provides a comprehensive range of text embeddings and reranking models in various sizes (0.6B, 4B, and 8B). This series inherits the exceptional multilingual capabilities, long-text understanding, and reasoning skills of its foundational model. The Qwen3 Embedding series represents significant advancements in multiple text embedding and ranking tasks, including text retrieval, code retrieval, text classification, text clustering, and bitext mining.

Exceptional Versatility: The embedding model has achieved state-of-the-art performance across a wide range of downstream application evaluations. The 8B size embedding model ranks No.1 in the MTEB multilingual leaderboard (as of June 5, 2025, score 70.58), while the reranking model excels in various text retrieval scenarios.

Comprehensive Flexibility: The Qwen3 Embedding series offers a full spectrum of sizes (from 0.6B to 8B) for both embedding and reranking models, catering to diverse use cases that prioritize efficiency and effectiveness. Developers can seamlessly combine these two modules. Additionally, the embedding model allows for flexible vector definitions across all dimensions, and both embedding and reranking models support user-defined instructions to enhance performance for specific tasks, languages, or scenarios.

Multilingual Capability: The Qwen3 Embedding series offer support for over 100 languages, thanks to the multilingual capabilities of Qwen3 models. This includes various programming languages, and provides robust multilingual, cross-lingual, and code retrieval capabilities.

Model Overview
Qwen3-Embedding-0.6B has the following features:

Model Type: Text Embedding
Supported Languages: 100+ Languages
Number of Parameters: 0.6B
Context Length: 32k
Embedding Dimension: Up to 1024, supports user-defined output dimensions ranging from 32 to 1024
For more details, including benchmark evaluation, hardware requirements, and inference performance, please refer to our blog, GitHub.

Qwen3 Embedding Series Model list
Model Type	Models	Size	Layers	Sequence Length	Embedding Dimension	MRL Support	Instruction Aware
Text Embedding	Qwen3-Embedding-0.6B	0.6B	28	32K	1024	Yes	Yes
Text Embedding	Qwen3-Embedding-4B	4B	36	32K	2560	Yes	Yes
Text Embedding	Qwen3-Embedding-8B	8B	36	32K	4096	Yes	Yes
Text Reranking	Qwen3-Reranker-0.6B	0.6B	28	32K	-	-	Yes
Text Reranking	Qwen3-Reranker-4B	4B	36	32K	-	-	Yes
Text Reranking	Qwen3-Reranker-8B	8B	36	32K	-	-	Yes
Note:

MRL Support indicates whether the embedding model supports custom dimensions for the final embedding.
Instruction Aware notes whether the embedding or reranking model supports customizing the input instruction according to different tasks.
Our evaluation indicates that, for most downstream tasks, using instructions (instruct) typically yields an improvement of 1% to 5% compared to not using them. Therefore, we recommend that developers create tailored instructions specific to their tasks and scenarios. In multilingual contexts, we also advise users to write their instructions in English, as most instructions utilized during the model training process were originally written in English.
""",
    ][:batch_size]
    input_ids = [tokenizer.encode(p) for p in prompts]
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(prompts)):
        print(f"--------batch {i} prefill len is {len(input_ids[i])} -----------")
        assert len(input_ids[i]) > bench_args.cut_len

        tmp_input_ids = input_ids[i][: bench_args.cut_len]
        req = Req(
            rid=i,
            origin_input_text=prompts[i],
            origin_input_ids=tmp_input_ids,
            sampling_params=sampling_params,
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req)

    return input_ids, reqs


def prepare_extend_inputs_for_correctness_test(
    bench_args, input_ids, reqs, model_runner
):
    for i in range(len(reqs)):
        req = reqs[i]
        req.fill_ids += input_ids[i][bench_args.cut_len :]
        req.prefix_indices = model_runner.req_to_token_pool.req_to_token[
            i, : bench_args.cut_len
        ]
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
    return reqs


def prepare_synthetic_inputs_for_latency_test(batch_size, input_len):
    input_ids = np.random.randint(0, 10000, (batch_size, input_len), dtype=np.int32)
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(input_ids)):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=list(input_ids[i]),
            sampling_params=sampling_params,
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req)

    return reqs


@torch.no_grad
def extend(reqs, model_runner):
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=None,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        enable_custom_logit_processor=False,
    )
    batch.prepare_for_extend()
    _maybe_prepare_dp_attn_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output, _ = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits, batch


@torch.no_grad
def decode(input_token_ids, batch, model_runner):
    batch.output_ids = input_token_ids
    batch.prepare_for_decode()
    _maybe_prepare_dp_attn_batch(batch, model_runner)
    _maybe_prepare_tbo_heto_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output, _ = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits


def _maybe_prepare_dp_attn_batch(batch: ScheduleBatch, model_runner):
    if model_runner.server_args.enable_dp_attention:
        Scheduler.prepare_dp_attn_batch_raw(
            batch,
            dp_size=model_runner.server_args.dp_size,
            attn_tp_size=1,
            moe_dense_tp_size=model_runner.server_args.moe_dense_tp_size,
            tp_cpu_group=model_runner.tp_group.cpu_group,
            get_idle_batch=None,
            disable_cuda_graph=model_runner.server_args.disable_cuda_graph,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            speculative_num_draft_tokens=None,
        )


def _maybe_prepare_tbo_heto_batch(batch: ScheduleBatch, model_runner):
    if (
        model_runner.server_args.enable_two_batch_overlap
        and model_runner.server_args.two_batch_overlap_mode == "heto"
        and batch.tbo_split_seq_index is None
    ):
        Scheduler.prepare_tbo_heto(
            batch, model_runner.server_args.two_batch_overlap_mode
        )


def correctness_test(
    server_args,
    port_args,
    bench_args,
    tp_rank,
):
    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, tp_rank)

    # Prepare inputs
    input_ids, reqs = prepare_inputs_for_correctness_test(
        bench_args, tokenizer, bench_args.batch_size[0]
    )
    rank_print(f"\n{input_ids=}\n")

    if bench_args.cut_len > 0:
        # Prefill
        next_token_ids, next_token_logits, batch = extend(reqs, model_runner)
        rank_print(f"prefill logits (first half): {next_token_logits} \n")

    # Prepare extend inputs
    reqs = prepare_extend_inputs_for_correctness_test(
        bench_args, input_ids, reqs, model_runner
    )

    # Extend (prefill w/ KV cache)
    next_token_ids, next_token_logits, batch = extend(reqs, model_runner)
    rank_print(f"prefill logits (final): {next_token_logits} \n")

    # Decode
    output_ids = [input_ids[i] + [next_token_ids[i]] for i in range(len(input_ids))]
    for _ in range(bench_args.output_len[0] - 1):
        next_token_ids, next_token_logits = decode(next_token_ids, batch, model_runner)
        next_token_ids_list = next_token_ids.tolist()
        for i in range(len(reqs)):
            output_ids[i].append(next_token_ids_list[i])
        rank_print(f"decode logits: {next_token_logits} \n")

    # Print output texts
    for i in range(len(reqs)):
        rank_print(f"========== Prompt {i} ==========")
        rank_print(tokenizer.decode(output_ids[i]), "\n")


def synchronize(device):
    torch.get_device_module(device).synchronize()


def latency_test_run_once(
    run_name,
    model_runner,
    rank_print,
    reqs,
    batch_size,
    input_len,
    output_len,
    device,
    log_decode_step,
    profile,
    profile_filename_prefix,
):
    max_batch_size = model_runner.max_total_num_tokens // (input_len + output_len)
    if batch_size > max_batch_size:
        rank_print(
            f"skipping ({batch_size}, {input_len}, {output_len}) due to max batch size limit"
        )
        return

    # Clear the pools.
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    measurement_results = {
        "run_name": run_name,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
    }

    tot_latency = 0

    profiler = None
    if profile:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_stack=True,
        )
        profiler.start()

    # Prefill
    synchronize(device)
    tic = time.perf_counter()
    next_token_ids, _, batch = extend(reqs, model_runner)
    synchronize(device)
    prefill_latency = time.perf_counter() - tic
    tot_latency += prefill_latency
    throughput = input_len * batch_size / prefill_latency
    rank_print(
        f"Prefill. latency: {prefill_latency:6.5f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["prefill_latency"] = prefill_latency
    measurement_results["prefill_throughput"] = throughput

    # Decode
    decode_latencies = []
    for i in range(output_len - 1):
        synchronize(device)
        tic = time.perf_counter()
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        synchronize(device)
        latency = time.perf_counter() - tic
        tot_latency += latency
        throughput = batch_size / latency
        decode_latencies.append(latency)
        if i < 5 or (log_decode_step > 0 and i % log_decode_step == 0):
            rank_print(
                f"Decode {i}. Batch size: {batch_size}, latency: {latency:6.5f} s, throughput: {throughput:9.2f} token/s"
            )

    if profile:
        profiler.stop()
        profile_filename = f"{profile_filename_prefix}_batch{batch_size}_input{input_len}_output{output_len}.trace.json.gz"
        parent_dir = os.path.dirname(os.path.abspath(profile_filename))
        os.makedirs(parent_dir, exist_ok=True)
        profiler.export_chrome_trace(profile_filename)
        rank_print(f"torch profiler chrome trace saved to {profile_filename}")

    # Record decode timing from 2nd output
    if output_len > 1:
        med_decode_latency = np.median(decode_latencies)
        med_decode_throughput = batch_size / med_decode_latency
        rank_print(
            f"Decode.  median latency: {med_decode_latency:6.5f} s, median throughput: {med_decode_throughput:9.2f} token/s"
        )
        measurement_results["median_decode_latency"] = med_decode_latency
        measurement_results["median_decode_throughput"] = med_decode_throughput

    throughput = (input_len + output_len) * batch_size / tot_latency
    rank_print(
        f"Total. latency: {tot_latency:6.3f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["total_latency"] = tot_latency
    measurement_results["overall_throughput"] = throughput
    return measurement_results


def latency_test(
    server_args,
    port_args,
    bench_args,
    tp_rank,
):
    # Set CPU affinity
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, tp_rank)

    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, tp_rank)

    # Prepare inputs for warm up
    reqs = prepare_synthetic_inputs_for_latency_test(
        bench_args.batch_size[0], bench_args.input_len[0]
    )

    # Warm up
    rank_print("Warmup ...")
    latency_test_run_once(
        bench_args.run_name,
        model_runner,
        rank_print,
        reqs,
        bench_args.batch_size[0],
        bench_args.input_len[0],
        min(32, bench_args.output_len[0]),  # shorter decoding to speed up the warmup
        server_args.device,
        log_decode_step=0,
        profile=False,
        profile_filename_prefix="",  # not used
    )

    rank_print("Benchmark ...")

    # Run the sweep
    result_list = []
    for bs, il, ol, _ in itertools.product(
        bench_args.batch_size,
        bench_args.input_len,
        bench_args.output_len,
        [1] * bench_args.iterations,
    ):
        reqs = prepare_synthetic_inputs_for_latency_test(bs, il)
        ret = latency_test_run_once(
            bench_args.run_name,
            model_runner,
            rank_print,
            reqs,
            bs,
            il,
            ol,
            server_args.device,
            bench_args.log_decode_step,
            bench_args.profile if tp_rank == 0 else None,
            bench_args.profile_filename_prefix,
        )
        if ret is not None:
            result_list.append(ret)

    # Write results in jsonlines format on rank 0.
    if tp_rank == 0 and bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            for result in result_list:
                fout.write(json.dumps(result) + "\n")

    if server_args.tp_size > 1:
        destroy_distributed_environment()


def main(server_args, bench_args):
    server_args.cuda_graph_max_bs = max(bench_args.batch_size)

    _set_envs_and_config(server_args)

    if server_args.model_path:
        if bench_args.correctness_test:
            work_func = correctness_test
        else:
            work_func = latency_test
    else:
        raise ValueError(
            "Provide --model-path for running the tests or "
            "provide --result-filename for plotting the results"
        )

    port_args = PortArgs.init_new(server_args)

    if server_args.tp_size == 1:
        work_func(server_args, port_args, bench_args, 0)
    else:
        workers = []
        for tp_rank in range(server_args.tp_size):
            proc = multiprocessing.Process(
                target=work_func,
                args=(
                    server_args,
                    port_args,
                    bench_args,
                    tp_rank,
                ),
            )
            proc.start()
            workers.append(proc)

        for proc in workers:
            proc.join()

        proc.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        main(server_args, bench_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)
