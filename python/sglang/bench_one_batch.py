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
        """There are two books, named <<First book>> and <<Second book>>, Pls do Summary and compare the content for these two books.
You must tell the difference between two books.
you will focus on identifying potential key themes, probable main plot points or central arguments, 
and the overall tone of each book. You will approach this task assuming a broad readership, 
creating summaries suitable for someone unfamiliar with the works. You will focusing on the core ideas and potential takeaways.
You'll do my best to glean the heart of each book and present it in a clear and helpful manner. 


<<First book>>
Harry Potter and the Sorcerer\'s Stone


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
half-moon glasses. "It would be enough to turn any boy\'s head. Famous
before he can walk and talk! Famous for something he won\'t even
remember! CarA you see how much better off he\'ll be, growing up away
from all that until he\'s ready to take it?"

Professor McGonagall opened her mouth, changed her mind, swallowed, and
then said, "Yes -- yes, you\'re right, of course. But how is the boy
getting here, Dumbledore?" She eyed his cloak suddenly as though she
thought he might be hiding Harry underneath it.

"Hagrid\'s bringing him."

"You think it -- wise -- to trust Hagrid with something as important as
this?"

I would trust Hagrid with my life," said Dumbledore.

"I\'m not saying his heart isn\'t in the right place," said Professor
McGonagall grudgingly, "but you can\'t pretend he\'s not careless. He does
tend to -- what was that?"

A low rumbling sound had broken the silence around them. It grew
steadily louder as they looked up and down the street for some sign of a
headlight; it swelled to a roar as they both looked up at the sky -- and
a huge motorcycle fell out of the air and landed on the road in front of
them.

If the motorcycle was huge, it was nothing to the man sitting astride
it. He was almost twice as tall as a normal man and at least five times
as wide. He looked simply too big to be allowed, and so wild - long
tangles of bushy black hair and beard hid most of his face, he had hands
the size of trash can lids, and his feet in their leather boots were
like baby dolphins. In his vast, muscular arms he was holding a bundle
of blankets.

"Hagrid," said Dumbledore, sounding relieved. "At last. And where did
you get that motorcycle?"

"Borrowed it, Professor Dumbledore, sit," said the giant, climbing
carefully off the motorcycle as he spoke. "Young Sirius Black lent it to
me. I\'ve got him, sir."

"No problems, were there?"

"No, sir -- house was almost destroyed, but I got him out all right
before the Muggles started swarmin\' around. He fell asleep as we was
flyin\' over Bristol."

Dumbledore and Professor McGonagall bent forward over the bundle of
blankets. Inside, just visible, was a baby boy, fast asleep. Under a
tuft of jet-black hair over his forehead they could see a curiously
shaped cut, like a bolt of lightning.

"Is that where -?" whispered Professor McGonagall.

"Yes," said Dumbledore. "He\'ll have that scar forever."

"Couldn\'t you do something about it, Dumbledore?"

"Even if I could, I wouldn\'t. Scars can come in handy. I have one myself
above my left knee that is a perfect map of the London Underground. Well
-- give him here, Hagrid -- we\'d better get this over with."

Dumbledore took Harry in his arms and turned toward the Dursleys\' house.

"Could I -- could I say good-bye to him, sir?" asked Hagrid. He bent his
great, shaggy head over Harry and gave him what must have been a very
scratchy, whiskery kiss. Then, suddenly, Hagrid let out a howl like a
wounded dog.

"Shhh!" hissed Professor McGonagall, "you\'ll wake the Muggles!"

"S-s-sorry," sobbed Hagrid, taking out a large, spotted handkerchief and
burying his face in it. "But I c-c-can\'t stand it -- Lily an\' James dead
-- an\' poor little Harry off ter live with Muggles -"

"Yes, yes, it\'s all very sad, but get a grip on yourself, Hagrid, or
we\'ll be found," Professor McGonagall whispered, patting Hagrid gingerly
on the arm as Dumbledore stepped over the low garden wall and walked to
the front door. He laid Harry gently on the doorstep, took a letter out
of his cloak, tucked it inside Harry\'s blankets, and then came back to
the other two. For a full minute the three of them stood and looked at
the little bundle; Hagrid\'s shoulders shook, Professor McGonagall
blinked furiously, and the twinkling light that usually shone from
Dumbledore\'s eyes seemed to have gone out.

"Well," said Dumbledore finally, "that\'s that. We\'ve no business staying
here. We may as well go and join the celebrations."

"Yeah," said Hagrid in a very muffled voice, "I\'ll be takin\' Sirius his
bike back. G\'night, Professor McGonagall -- Professor Dumbledore, sir."

Wiping his streaming eyes on his jacket sleeve, Hagrid swung himself
onto the motorcycle and kicked the engine into life; with a roar it rose
into the air and off into the night.

"I shall see you soon, I expect, Professor McGonagall," said Dumbledore,
nodding to her. Professor McGonagall blew her nose in reply.

Dumbledore turned and walked back down the street. On the corner he
stopped and took out the silver Put-Outer. He clicked it once, and
twelve balls of light sped back to their street lamps so that Privet
Drive glowed suddenly orange and he could make out a tabby cat slinking
around the corner at the other end of the street. He could just see the
bundle of blankets on the step of number four.

"Good luck, Harry," he murmured. He turned on his heel and with a swish
of his cloak, he was gone.

A breeze ruffled the neat hedges of Privet Drive, which lay silent and
tidy under the inky sky, the very last place you would expect
astonishing things to happen. Harry Potter rolled over inside his
blankets without waking up. One small hand closed on the letter beside
him and he slept on, not knowing he was special, not knowing he was
famous, not knowing he would be woken in a few hours\' time by Mrs.
Dursley\'s scream as she opened the front door to put out the milk
bottles, nor that he would spend the next few weeks being prodded and
pinched by his cousin Dudley... He couldn\'t know that at this very
moment, people meeting in secret all over the country were holding up
their glasses and saying in hushed voices: "To Harry Potter -- the boy
who lived!"

"S-s-sorry," sobbed Hagrid, taking out a large, spotted handkerchief and
burying his face in it. "But I c-c-can\'t stand it -- Lily an\' James dead
-- an\' poor little Harry off ter live with Muggles -"

"Yes, yes, it\'s all very sad, but get a grip on yourself, Hagrid, or
we\'ll be found," Professor McGonagall whispered, patting Hagrid gingerly
on the arm as Dumbledore stepped over the low garden wall and walked to
the front door. He laid Harry gently on the doorstep, took a letter out
of his cloak, tucked it inside Harry\'s blankets, and then came back to
the other two. For a full minute the three of them stood and looked at
the little bundle; Hagrid\'s shoulders shook, Professor McGonagall
blinked furiously, and the twinkling light that usually shone from
Dumbledore\'s eyes seemed to have gone out.

"Well," said Dumbledore finally, "that\'s that. We\'ve no business staying
here. We may as well go and join the celebrations."

"Yeah," said Hagrid in a very muffled voice, "I\'ll be takin\' Sirius his
bike back. G\'night, Professor McGonagall -- Professor Dumbledore, sir."

Wiping his streaming eyes on his jacket sleeve, Hagrid swung himself
onto the motorcycle and kicked the engine into life; with a roar it rose
into the air and off into the night.

"I shall see you soon, I expect, Professor McGonagall," said Dumbledore,
nodding to her. Professor McGonagall blew her nose in reply.

Dumbledore turned and walked back down the street. On the corner he
stopped and took out the silver Put-Outer. He clicked it once, and
twelve balls of light sped back to their street lamps so that Privet
Drive glowed suddenly orange and he could make out a tabby cat slinking
around the corner at the other end of the street. He could just see the
bundle of blankets on the step of number four.

"Good luck, Harry," he murmured. He turned on his heel and with a swish
of his cloak, he was gone.



CHAPTER TWO

THE VANISHING GLASS

Nearly ten years had passed since the Dursleys had woken up to find
their nephew on the front step, but Privet Drive had hardly changed at
all. The sun rose on the same tidy front gardens and lit up the brass
number four on the Dursleys\' front door; it crept into their living
room, which was almost exactly the same as it had been on the night when
Mr. Dursley had seen that fateful news report about the owls. Only the
photographs on the mantelpiece really showed how much time had passed.
Ten years ago, there had been lots of pictures of what looked like a
large pink beach ball wearing different-colored bonnets -- but Dudley
Dursley was no longer a baby, and now the photographs showed a large
blond boy riding his first bicycle, on a carousel at the fair, playing a
computer game with his father, being hugged and kissed by his mother.
The room held no sign at all that another boy lived in the house, too.

Yet Harry Potter was still there, asleep at the moment, but not for
long. His Aunt Petunia was awake and it was her shrill voice that made
the first noise of the day.

"Up! Get up! Now!"

Harry woke with a start. His aunt rapped on the door again.

"Up!" she screeched. Harry heard her walking toward the kitchen and then
the sound of the frying pan being put on the stove. He rolled onto his
back and tried to remember the dream he had been having. It had been a
good one. There had been a flying motorcycle in it. He had a funny
feeling he\'d had the same dream before.

His aunt was back outside the door.

"Are you up yet?" she demanded.

"Nearly," said Harry.

"Well, get a move on, I want you to look after the bacon. And don\'t you
dare let it burn, I want everything perfect on Duddy\'s birthday."

Harry groaned.

"What did you say?" his aunt snapped through the door.

"Nothing, nothing..."

Dudley\'s birthday -- how could he have forgotten? Harry got slowly out
of bed and started looking for socks. He found a pair under his bed and,
after pulling a spider off one of them, put them on. Harry was used to
spiders, because the cupboard under the stairs was full of them, and
that was where he slept.

When he was dressed he went down the hall into the kitchen. The table
was almost hidden beneath all Dudley\'s birthday presents. It looked as
though Dudley had gotten the new computer he wanted, not to mention the
second television and the racing bike. Exactly why Dudley wanted a
racing bike was a mystery to Harry, as Dudley was very fat and hated
exercise -- unless of course it involved punching somebody. Dudley\'s
favorite punching bag was Harry, but he couldn\'t often catch him. Harry
didn\'t look it, but he was very fast.

Perhaps it had something to do with living in a dark cupboard, but Harry
had always been small and skinny for his age. He looked even smaller and
skinnier than he really was because all he had to wear were old clothes
of Dudley\'s, and Dudley was about four times bigger than he was. Harry
had a thin face, knobbly knees, black hair, and bright green eyes. He
wore round glasses held together with a lot of Scotch tape because of
all the times Dudley had punched him on the nose. The only thing Harry
liked about his own appearance was a very thin scar on his forehead that
was shaped like a bolt of lightning. He had had it as long as he could
remember, and the first question he could ever remember asking his Aunt
Petunia was how he had gotten it.

"In the car crash when your parents died," she had said. "And don\'t ask
questions."

Don\'t ask questions -- that was the first rule for a quiet life with the
Dursleys.

Uncle Vernon entered the kitchen as Harry was turning over the bacon.

"Comb your hair!" he barked, by way of a morning greeting.

About once a week, Uncle Vernon looked over the top of his newspaper and
shouted that Harry needed a haircut. Harry must have had more haircuts
than the rest of the boys in his class put

together, but it made no difference, his hair simply grew that way --
all over the place.

Harry was frying eggs by the time Dudley arrived in the kitchen with his
mother. Dudley looked a lot like Uncle Vernon. He had a large pink face,
not much neck, small, watery blue eyes, and thick blond hair that lay
smoothly on his thick, fat head. Aunt Petunia often said that Dudley
looked like a baby angel -- Harry often said that Dudley looked like a
pig in a wig.

Harry put the plates of egg and bacon on the table, which was difficult
as there wasn\'t much room. Dudley, meanwhile, was counting his presents.
His face fell.

"Thirty-six," he said, looking up at his mother and father. "That\'s two
less than last year."

"Darling, you haven\'t counted Auntie Marge\'s present, see, it\'s here
under this big one from Mommy and Daddy."

"All right, thirty-seven then," said Dudley, going red in the face.
Harry, who could see a huge Dudley tantrum coming on, began wolfing down
his bacon as fast as possible in case Dudley turned the table over.

Aunt Petunia obviously scented danger, too, because she said quickly,
"And we\'ll buy you another two presents while we\'re out today. How\'s
that, popkin? Two more presents. Is that all right\'\'

Dudley thought for a moment. It looked like hard work. Finally he said
slowly, "So I\'ll have thirty ... thirty..."

"Thirty-nine, sweetums," said Aunt Petunia.

"Oh." Dudley sat down heavily and grabbed the nearest parcel. "All right
then."

Uncle Vernon chuckled. "Little tyke wants his money\'s worth, just like
his father. \'Atta boy, Dudley!" He ruffled Dudley\'s hair.

At that moment the telephone rang and Aunt Petunia went to answer it
while Harry and Uncle Vernon watched Dudley unwrap the racing bike, a
video camera, a remote control airplane, sixteen new computer games, and
a VCR. He was ripping the paper off a gold wristwatch when Aunt Petunia
came back from the telephone looking both angry and worried.

"Bad news, Vernon," she said. "Mrs. Figg\'s broken her leg. She can\'t
take him." She jerked her head in Harry\'s direction.

Dudley\'s mouth fell open in horror, but Harry\'s heart gave a leap. Every
year on Dudley\'s birthday, his parents took him and a friend out for the
day, to adventure parks, hamburger restaurants, or the movies. Every
year, Harry was left behind with Mrs. Figg, a mad old lady who lived two
streets away. Harry hated it there. The whole house smelled of cabbage
and Mrs. Figg made him look at photographs of all the cats she\'d ever
owned.

"Now what?" said Aunt Petunia, looking furiously at Harry as though he\'d
planned this. Harry knew he ought to feel sorry that Mrs. Figg had
broken her leg, but it wasn\'t easy when he reminded himself it would be
a whole year before he had to look at Tibbles, Snowy, Mr. Paws, and
Tufty again.

"We could phone Marge," Uncle Vernon suggested.

"Don\'t be silly, Vernon, she hates the boy."

The Dursleys often spoke about Harry like this, as though he wasn\'t
there -- or rather, as though he was something very nasty that couldn\'t
understand them, like a slug.

"What about what\'s-her-name, your friend -- Yvonne?"

"On vacation in Majorca," snapped Aunt Petunia.

"You could just leave me here," Harry put in hopefully (he\'d be able to
watch what he wanted on television for a change and maybe even have a go
on Dudley\'s computer).

Aunt Petunia looked as though she\'d just swallowed a lemon.

"And come back and find the house in ruins?" she snarled.

"I won\'t blow up the house," said Harry, but they weren\'t listening.

"I suppose we could take him to the zoo," said Aunt Petunia slowly, "...
and leave him in the car...."

"That car\'s new, he\'s not sitting in it alone...."

Dudley began to cry loudly. In fact, he wasn\'t really crying -- it had
been years since he\'d really cried -- but he knew that if he screwed up
his face and wailed, his mother would give him anything he wanted.

"Dinky Duddydums, don\'t cry, Mummy won\'t let him spoil your special
day!" she cried, flinging her arms around him.

"I... don\'t... want... him... t-t-to come!" Dudley yelled between huge,
pretend sobs. "He always sp- spoils everything!" He shot Harry a nasty
grin through the gap in his mother\'s arms.

Just then, the doorbell rang -- "Oh, good Lord, they\'re here!" said Aunt
Petunia frantically -- and a moment later, Dudley\'s best friend, Piers
Polkiss, walked in with his mother. Piers was a scrawny boy with a face
like a rat. He was usually the one who held people\'s arms behind their
backs while Dudley hit them. Dudley stopped pretending to cry at once.

Half an hour later, Harry, who couldn\'t believe his luck, was sitting in
the back of the Dursleys\' car with Piers and Dudley, on the way to the
zoo for the first time in his life. His aunt and uncle hadn\'t been able
to think of anything else to do with him, but before they\'d left, Uncle
Vernon had taken Harry aside.

"I\'m warning you," he had said, putting his large purple face right up
close to Harry\'s, "I\'m warning you now, boy -- any funny business,
anything at all -- and you\'ll be in that cupboard from now until
Christmas."

"I\'m not going to do anything," said Harry, "honestly..

But Uncle Vernon didn\'t believe him. No one ever did.

The problem was, strange things often happened around Harry and it was
just no good telling the Dursleys he didn\'t make them happen.

Once, Aunt Petunia, tired of Harry coming back from the barbers looking
as though he hadn\'t been at all, had taken a pair of kitchen scissors
and cut his hair so short he was almost bald except for his bangs, which
she left "to hide that horrible scar." Dudley had laughed himself silly
at Harry, who spent a sleepless night imagining school the next day,
where he was already laughed at for his baggy clothes and taped glasses.
Next morning, however, he had gotten up to find his hair exactly as it
had been before Aunt Petunia had sheared it off He had been given a week
in his cupboard for this, even though he had tried to explain that he
couldn\'t explain how it had grown back so quickly.

Another time, Aunt Petunia had been trying to force him into a revolting
old sweater of Dudley\'s (brown with orange puff balls) -- The harder she
tried to pull it over his head, the smaller it seemed to become, until
finally it might have fitted a hand puppet, but certainly wouldn\'t fit
Harry. Aunt Petunia had decided it must have shrunk in the wash and, to
his great relief, Harry wasn\'t punished.

On the other hand, he\'d gotten into terrible trouble for being found on
the roof of the school kitchens. Dudley\'s gang had been chasing him as
usual when, as much to Harry\'s surprise as anyone else\'s, there he was
sitting on the chimney. The Dursleys had received a very angry letter
from Harry\'s headmistress telling them Harry had been climbing school
buildings. But all he\'d tried to do (as he shouted at Uncle Vernon
through the locked door of his cupboard) was jump behind the big trash
cans outside the kitchen doors. Harry supposed that the wind must have
caught him in mid- jump.

But today, nothing was going to go wrong. It was even worth being with
Dudley and Piers to be spending the day somewhere that wasn\'t school,
his cupboard, or Mrs. Figg\'s cabbage-smelling living room.

While he drove, Uncle Vernon complained to Aunt Petunia. He liked to
complain about things: people at work, Harry, the council, Harry, the
bank, and Harry were just a few of his favorite subjects. This morning,
it was motorcycles.

"... roaring along like maniacs, the young hoodlums," he said, as a
motorcycle overtook them.

I had a dream about a motorcycle," said Harry, remembering suddenly. "It
was flying."

Uncle Vernon nearly crashed into the car in front. He turned right
around in his seat and yelled at Harry, his face like a gigantic beet
with a mustache: "MOTORCYCLES DON\'T FLY!"

Dudley and Piers sniggered.

I know they don\'t," said Harry. "It was only a dream."

But he wished he hadn\'t said anything. If there was one thing the
Dursleys hated even more than his asking questions, it was his talking
about anything acting in a way it shouldn\'t, no matter if it was in a
dream or even a cartoon -- they seemed to think he might get dangerous
ideas.

It was a very sunny Saturday and the zoo was crowded with families. The
Dursleys bought Dudley and Piers large chocolate ice creams at the
entrance and then, because the smiling lady in the van had asked Harry
what he wanted before they could hurry him away, they bought him a cheap
lemon ice pop. It wasn\'t bad, either, Harry thought, licking it as they
watched a gorilla scratching its head who looked remarkably like Dudley,
except that it wasn\'t blond.

Harry had the best morning he\'d had in a long time. He was careful to
walk a little way apart from the Dursleys so that Dudley and Piers, who
were starting to get bored with the animals by lunchtime, wouldn\'t fall
back on their favorite hobby of hitting him. They ate in the zoo
restaurant, and when Dudley had a tantrum because his knickerbocker
glory didn\'t have enough ice cream on top, Uncle Vernon bought him
another one and Harry was allowed to finish the first.

Harry felt, afterward, that he should have known it was all too good to
last.

After lunch they went to the reptile house. It was cool and dark in
there, with lit windows all along the walls. Behind the glass, all sorts
of lizards and snakes were crawling and slithering over bits of wood and
stone. Dudley and Piers wanted to see huge, poisonous cobras and thick,
man-crushing pythons. Dudley quickly found the largest snake in the
place. It could have wrapped its body twice around Uncle Vernon\'s car
and crushed it into a trash can -- but at the moment it didn\'t look in
the mood. In fact, it was fast asleep.

Dudley stood with his nose pressed against the glass, staring at the
glistening brown coils.

"Make it move," he whined at his father. Uncle Vernon tapped on the
glass, but the snake didn\'t budge.

"Do it again," Dudley ordered. Uncle Vernon rapped the glass smartly
with his knuckles, but the snake just snoozed on.

"This is boring," Dudley moaned. He shuffled away.

Harry moved in front of the tank and looked intently at the snake. He
wouldn\'t have been surprised if it had died of boredom itself -- no
company except stupid people drumming their fingers on the glass trying
to disturb it all day long. It was worse than having a cupboard as a
bedroom, where the only visitor was Aunt Petunia hammering on the door
to wake you up; at least he got to visit the rest of the house.

The snake suddenly opened its beady eyes. Slowly, very slowly, it raised
its head until its eyes were on a level with Harry\'s.

It winked.

Harry stared. Then he looked quickly around to see if anyone was
watching. They weren\'t. He looked back at the snake and winked, too.

The snake jerked its head toward Uncle Vernon and Dudley, then raised
its eyes to the ceiling. It gave Harry a look that said quite plainly:

"I get that all the time.

"I know," Harry murmured through the glass, though he wasn\'t sure the
snake could hear him. "It must be really annoying."

The snake nodded vigorously.

"Where do you come from, anyway?" Harry asked.

The snake jabbed its tail at a little sign next to the glass. Harry
peered at it.

Boa Constrictor, Brazil.

"Was it nice there?"

The boa constrictor jabbed its tail at the sign again and Harry read on:
This specimen was bred in the zoo. "Oh, I see -- so you\'ve never been to
Brazil?"

As the snake shook its head, a deafening shout behind Harry made both of
them jump.

"DUDLEY! MR. DURSLEY! COME AND LOOK AT THIS SNAKE! YOU WON\'T BELIEVE
WHAT IT\'S DOING!"

Dudley came waddling toward them as fast as he could.

"Out of the way, you," he said, punching Harry in the ribs. Caught by
surprise, Harry fell hard on the concrete floor. What came next happened
so fast no one saw how it happened -- one second, Piers and Dudley were
leaning right up close to the glass, the next, they had leapt back with
howls of horror.

Harry sat up and gasped; the glass front of the boa constrictor\'s tank
had vanished. The great snake was uncoiling itself rapidly, slithering
out onto the floor. People throughout the reptile house screamed and
started running for the exits.

As the snake slid swiftly past him, Harry could have sworn a low,
hissing voice said, "Brazil, here I come.... Thanksss, amigo."

The keeper of the reptile house was in shock.

"But the glass," he kept saying, "where did the glass go?"

The zoo director himself made Aunt Petunia a cup of strong, sweet tea
while he apologized over and over again. Piers and Dudley could only
gibber. As far as Harry had seen, the snake hadn\'t done anything except
snap playfully at their heels as it passed, but by the time they were
all back in Uncle Vernon\'s car, Dudley was telling them how it had
nearly bitten off his leg, while Piers was swearing it had tried to
squeeze him to death. But worst of all, for Harry at least, was Piers
calming down enough to say, "Harry was talking to it, weren\'t you,
Harry?"

Uncle Vernon waited until Piers was safely out of the house before
starting on Harry. He was so angry he could hardly speak. He managed to
say, "Go -- cupboard -- stay -- no meals," before he collapsed into a
chair, and Aunt Petunia had to run and get him a large brandy.

Harry lay in his dark cupboard much later, wishing he had a watch. He
didn\'t know what time it was and he couldn\'t be sure the Dursleys were
asleep yet. Until they were, he couldn\'t risk sneaking to the kitchen
for some food.

He\'d lived with the Dursleys almost ten years, ten miserable years, as
long as he could remember, ever since he\'d been a baby and his parents
had died in that car crash. He couldn\'t remember being in the car when
his parents had died. Sometimes, when he strained his memory during long
hours in his cupboard, he came up with a strange vision: a blinding
flash of green light and a burn- ing pain on his forehead. This, he
supposed, was the crash, though he couldn\'t imagine where all the green
light came from. He couldn\'t remember his parents at all. His aunt and
uncle never spoke about them, and of course he was forbidden to ask
questions. There were no photographs of them in the house.

When he had been younger, Harry had dreamed and dreamed of some unknown
relation coming take him away, but it had never happened; the
Dursleys were his only family. Yet sometimes he thought (or maybe hoped)
that strangers in the street seemed to know him. Very strange strangers
they were, too. A tiny man in a violet top hat had bowed to him once
while out shopping with Aunt Petunia and Dudley. After asking Harry
furiously if he knew the man, Aunt Petunia had rushed them out of the
shop without buying anything. A wild-looking old woman dressed all in
green had waved merrily at him once on a bus. A bald man in a very long
purple coat had actually shaken his hand in the street the other day and
then walked away without a word. The weirdest thing about all these
people was the way they seemed to vanish the second Harry tried to get a
closer look.

At school, Harry had no one. Everybody knew that Dudley\'s gang hated
that odd Harry Potter in his baggy old clothes and broken glasses, and
nobody liked to disagree with Dudley\'s gang.


CHAPTER THREE

THE LETTERS FROM NO ONE

The escape of the Brazilian boa constrictor earned Harry his
longest-ever punishment. By the time he was allowed out of his cupboard
again, the summer holidays had started and Dudley had already broken his
new video camera, crashed his remote control airplane, and, first time
out on his racing bike, knocked down old Mrs. Figg as she crossed Privet
Drive on her crutches.

Harry was glad school was over, but there was no escaping Dudley\'s gang,
who visited the house every single day. Piers, Dennis, Malcolm, and
Gordon were all big and stupid, but as Dudley was the biggest and
stupidest of the lot, he was the leader. The rest of them were all quite
happy to join in Dudley\'s favorite sport: Harry Hunting.

This was why Harry spent as much time as possible out of the house,
wandering around and thinking about the end of the holidays, where he
could see a tiny ray of hope. When September came he would be going off
to secondary school and, for the first time in his life, he wouldn\'t be
with Dudley. Dudley had been accepted at Uncle Vernon\'s old private
school, Smeltings. Piers Polkiss was going there too. Harry, on the
other hand, was going to Stonewall High, the local public school. Dudley
thought this was very funny.

"They stuff people\'s heads down the toilet the first day at Stonewall,"
he told Harry. "Want to come upstairs and practice?"

"No, thanks," said Harry. "The poor toilet\'s never had anything as
horrible as your head down it -- it might be sick." Then he ran, before
Dudley could work out what he\'d said.

One day in July, Aunt Petunia took Dudley to London to buy his Smeltings
uniform, leaving Harry at Mrs. Figg\'s. Mrs. Figg wasn \'t as bad as
usual. It turned out she\'d broken her leg tripping over one of her cats,
and she didn\'t seem quite as fond of them as before. She let Harry watch
television and gave him a bit of chocolate cake that tasted as though
she\'d had it for several years.

That evening, Dudley paraded around the living room for the family in
his brand-new uniform. Smeltings\' boys wore maroon tailcoats, orange
knickerbockers, and flat straw hats called boaters. They also carried
knobbly sticks, used for hitting each other while the teachers weren\'t
looking. This was supposed to be good training for later life.

As he looked at Dudley in his new knickerbockers, Uncle Vernon said
gruffly that it was the proudest moment of his life. Aunt Petunia burst
into tears and said she couldn\'t believe it was her Ickle Dudleykins, he
looked so handsome and grown-up. Harry didn\'t trust himself to speak. He
thought two of his ribs might already have cracked from trying not to
laugh.

There was a horrible smell in the kitchen the next morning when Harry
went in for breakfast. It seemed to be coming from a large metal tub in
the sink. He went to have a look. The tub was full of what looked like
dirty rags swimming in gray water.

"What\'s this?" he asked Aunt Petunia. Her lips tightened as they always
did if he dared to ask a question.

"Your new school uniform," she said.

Harry looked in the bowl again.

"Oh," he said, "I didn\'t realize it had to be so wet."

"DotA be stupid," snapped Aunt Petunia. "I\'m dyeing some of Dudley\'s old
things gray for you. It\'ll look just like everyone else\'s when I\'ve
finished."

Harry seriously doubted this, but thought it best not to argue. He sat
down at the table and tried not to think about how he was going to look
on his first day at Stonewall High -- like he was wearing bits of old
elephant skin, probably.

Dudley and Uncle Vernon came in, both with wrinkled noses because of the
smell from Harry\'s new uniform. Uncle Vernon opened his newspaper as
usual and Dudley banged his Smelting stick, which he carried everywhere,
on the table.

They heard the click of the mail slot and flop of letters on the
doormat.

"Get the mail, Dudley," said Uncle Vernon from behind his paper.

"Make Harry get it."

"Get the mail, Harry."

"Make Dudley get it."

"Poke him with your Smelting stick, Dudley."

Harry dodged the Smelting stick and went to get the mail. Three things
lay on the doormat: a postcard from Uncle Vernon\'s sister Marge, who was
vacationing on the Isle of Wight, a brown envelope that looked like a
bill, and -- a letter for Harry.

Harry picked it up and stared at it, his heart twanging like a giant
elastic band. No one, ever, in his whole life, had written to him. Who
would? He had no friends, no other relatives -- he didn\'t belong to the
library, so he\'d never even got rude notes asking for books back. Yet
here it was, a letter, addressed so plainly there could be no mistake:

Mr. H. Potter

The Cupboard under the Stairs

4 Privet Drive

Little Whinging

Surrey

The envelope was thick and heavy, made of yellowish parchment, and the
address was written in emerald-green ink. There was no stamp.

Turning the envelope over, his hand trembling, Harry saw a purple wax
seal bearing a coat of arms; a lion, an eagle, a badger, and a snake
surrounding a large letter H.

"Hurry up, boy!" shouted Uncle Vernon from the kitchen. "What are you
doing, checking for letter bombs?" He chuckled at his own joke.

Harry went back to the kitchen, still staring at his letter. He handed
Uncle Vernon the bill and the postcard, sat down, and slowly began to
open the yellow envelope.

Uncle Vernon ripped open the bill, snorted in disgust, and flipped over
the postcard.

"Marge\'s ill," he informed Aunt Petunia. "Ate a funny whelk. --."

"Dad!" said Dudley suddenly. "Dad, Harry\'s got something!"

Harry was on the point of unfolding his letter, which was written on the
same heavy parchment as the envelope, when it was jerked sharply out of
his hand by Uncle Vernon.

"That\'s mine!" said Harry, trying to snatch it back.

"Who\'d be writing to you?" sneered Uncle Vernon, shaking the letter open
with one hand and glancing at it. His face went from red to green faster
than a set of traffic lights. And it didn\'t stop there. Within seconds
it was the grayish white of old porridge.

"P-P-Petunia!" he gasped.

Dudley tried to grab the letter to read it, but Uncle Vernon held it
high out of his reach. Aunt Petunia took it curiously and read the first
line. For a moment it looked as though she might faint. She clutched her
throat and made a choking noise.

"Vernon! Oh my goodness -- Vernon!"

They stared at each other, seeming to have forgotten that Harry and
Dudley were still in the room. Dudley wasn\'t used to being ignored. He
gave his father a sharp tap on the head with his Smelting stick.

"I want to read that letter," he said loudly. want to read it," said
Harry furiously, "as it\'s mine."

"Get out, both of you," croaked Uncle Vernon, stuffing the letter back
inside its envelope.

Harry didn\'t move.

I WANT MY LETTER!" he shouted.

"Let me see it!" demanded Dudley.

"OUT!" roared Uncle Vernon, and he took both Harry and Dudley by the
scruffs of their necks and threw them into the hall, slamming the
kitchen door behind them. Harry and Dudley promptly had a furious but
silent fight over who would listen at the keyhole; Dudley won, so Harry,
his glasses dangling from one ear, lay flat on his stomach to listen at
the crack between door and floor.

"Vernon," Aunt Petunia was saying in a quivering voice, "look at the
address -- how could they possibly know where he sleeps? You don\'t think
they\'re watching the house?"

"Watching -- spying -- might be following us," muttered Uncle Vernon
wildly.

"But what should we do, Vernon? Should we write back? Tell them we don\'t
want --"

Harry could see Uncle Vernon\'s shiny black shoes pacing up and down the
kitchen.

"No," he said finally. "No, we\'ll ignore it. If they don\'t get an
answer... Yes, that\'s best... we won\'t do anything....

"But --"

"I\'m not having one in the house, Petunia! Didn\'t we swear when we took
him in we\'d stamp out that dangerous nonsense?"

That evening when he got back from work, Uncle Vernon did something he\'d
never done before; he visited Harry in his cupboard.

"Where\'s my letter?" said Harry, the moment Uncle Vernon had squeezed
through the door. "Who\'s writing to me?"

"No one. it was addressed to you by mistake," said Uncle Vernon shortly.
"I have burned it."

"It was not a mistake," said Harry angrily, "it had my cupboard on it."

"SILENCE!" yelled Uncle Vernon, and a couple of spiders fell from the
ceiling. He took a few deep breaths and then forced his face into a
smile, which looked quite painful.

"Er -- yes, Harry -- about this cupboard. Your aunt and I have been
thinking... you\'re really getting a bit big for it... we think it might
be nice if you moved into Dudley\'s second bedroom.

"Why?" said Harry.

"Don\'t ask questions!" snapped his uncle. "Take this stuff upstairs,
now."

The Dursleys\' house had four bedrooms: one for Uncle Vernon and Aunt
Petunia, one for visitors (usually Uncle Vernon\'s sister, Marge), one
where Dudley slept, and one where Dudley kept all the toys and things
that wouldn\'t fit into his first bedroom. It only took Harry one trip
upstairs to move everything he owned from the cupboard to this room. He
sat down on the bed and stared around him. Nearly everything in here was
broken. The month-old video camera was lying on top of a small, working
tank Dudley had once driven over the next door neighbor\'s dog; in the
corner was Dudley\'s first-ever television set, which he\'d put his foot
through when his favorite program had been canceled; there was a large
birdcage, which had once held a parrot that Dudley had swapped at school
for a real air rifle, which was up on a shelf with the end all bent
because Dudley had sat on it. Other shelves were full of books. They
were the only things in the room that looked as though they\'d never been
touched.

From downstairs came the sound of Dudley bawling at his mother, I don\'t
want him in there... I need that room... make him get out...."

Harry sighed and stretched out on the bed. Yesterday he\'d have given
anything to be up here. Today he\'d rather be back in his cupboard with
that letter than up here without it.

Next morning at breakfast, everyone was rather quiet. Dudley was in
shock. He\'d screamed, whacked his father with his Smelting stick, been
sick on purpose, kicked his mother, and thrown his tortoise through the
greenhouse roof, and he still didn\'t have his room back. Harry was
thinking about this time yesterday and bitterly wishing he\'d opened the
letter in the hall. Uncle Vernon and Aunt Petunia kept looking at each
other darkly.

When the mail arrived, Uncle Vernon, who seemed to be trying to be nice
to Harry, made Dudley go and get it. They heard him banging things with
his Smelting stick all the way down the hall. Then he shouted, "There\'s
another one! \'Mr. H. Potter, The Smallest Bedroom, 4 Privet Drive --\'"

With a strangled cry, Uncle Vernon leapt from his seat and ran down the
hall, Harry right behind him. Uncle Vernon had to wrestle Dudley to the
ground to get the letter from him, which was made difficult by the fact
that Harry had grabbed Uncle Vernon around the neck from behind. After a
minute of confused fighting, in which everyone got hit a lot by the
Smelting stick, Uncle Vernon straightened up, gasping for breath, with
Harry\'s letter clutched in his hand.

"Go to your cupboard -- I mean, your bedroom," he wheezed at Harry.
"Dudley -- go -- just go."

Harry walked round and round his new room. Someone knew he had moved out
of his cupboard and they seemed to know he hadn\'t received his first
letter. Surely that meant they\'d try again? And this time he\'d make sure
they didn\'t fail. He had a plan.

The repaired alarm clock rang at six o\'clock the next morning. Harry
turned it off quickly and dressed silently. He mustn\'t wake the
Dursleys. He stole downstairs without turning on any of the lights.

He was going to wait for the postman on the corner of Privet Drive and
get the letters for number four first. His heart hammered as he crept
across the dark hall toward the front door --

Harry leapt into the air; he\'d trodden on something big and squashy on
the doormat -- something alive!

Lights clicked on upstairs and to his horror Harry realized that the
big, squashy something had been his uncle\'s face. Uncle Vernon had been
lying at the foot of the front door in a sleeping bag, clearly making
sure that Harry didn\'t do exactly what he\'d been trying to do. He
shouted at Harry for about half an hour and then told him to go and make
a cup of tea. Harry shuffled miserably off into the kitchen and by the
time he got back, the mail had arrived, right into Uncle Vernon\'s lap.
Harry could see three letters addressed in green ink.

I want --" he began, but Uncle Vernon was tearing the letters into
pieces before his eyes. Uncle Vernon didnt go to work that day. He
stayed at home and nailed up the mail slot.

"See," he explained to Aunt Petunia through a mouthful of nails, "if
they can\'t deliver them they\'ll just give up."

"I\'m not sure that\'ll work, Vernon."

"Oh, these people\'s minds work in strange ways, Petunia, they\'re not
like you and me," said Uncle Vernon, trying to knock in a nail with the
piece of fruitcake Aunt Petunia had just brought him.

On Friday, no less than twelve letters arrived for Harry. As they
couldn\'t go through the mail slot they had been pushed under the door,
slotted through the sides, and a few even forced through the small
window in the downstairs bathroom.

Uncle Vernon stayed at home again. After burning all the letters, he got
out a hammer and nails and boarded up the cracks around the front and
back doors so no one could go out. He hummed "Tiptoe Through the Tulips"
as he worked, and jumped at small noises.

On Saturday, things began to get out of hand. Twenty-four letters to
Harry found their way into the house, rolled up and hidden inside each
of the two dozen eggs that their very confused milkman had handed Aunt
Petunia through the living room window. While Uncle Vernon made furious
telephone calls to the post office and the dairy trying to find someone
to complain to, Aunt Petunia shredded the letters in her food processor.

"Who on earth wants to talk to you this badly?" Dudley asked Harry in
amazement.

On Sunday morning, Uncle Vernon sat down at the breakfast table looking
tired and rather ill, but happy.

"No post on Sundays," he reminded them cheerfully as he spread marmalade
on his newspapers, "no damn letters today --"

Something came whizzing down the kitchen chimney as he spoke and caught
him sharply on the back of the head. Next moment, thirty or forty
letters came pelting out of the fireplace like bullets. The Dursleys
ducked, but Harry leapt into the air trying to catch one.

"Out! OUT!"

Uncle Vernon seized Harry around the waist and threw him into the hall.
When Aunt Petunia and Dudley had run out with their arms over their
faces, Uncle Vernon slammed the door shut. They could hear the letters
still streaming into the room, bouncing off the walls and floor.

"That does it," said Uncle Vernon, trying to speak calmly but pulling
great tufts out of his mustache at the same time. I want you all back
here in five minutes ready to leave. We\'re going away. Just pack some
clothes. No arguments!"

He looked so dangerous with half his mustache missing that no one dared
argue. Ten minutes later they had wrenched their way through the
boarded-up doors and were in the car, speeding toward the highway.
Dudley was sniffling in the back seat; his father had hit him round the
head for holding them up while he tried to pack his television, VCR, and
computer in his sports bag.

They drove. And they drove. Even Aunt Petunia didn\'t dare ask where they
were going. Every now and then Uncle Vernon would take a sharp turn and
drive in the opposite direction for a while. "Shake\'em off... shake \'em
off," he would mutter whenever he did this.

They didn\'t stop to eat or drink all day. By nightfall Dudley was
howling. He\'d never had such a bad day in his life. He was hungry, he\'d
missed five television programs he\'d wanted to see, and he\'d never gone
so long without blowing up an alien on his computer.

Uncle Vernon stopped at last outside a gloomy-looking hotel on the
outskirts of a big city. Dudley and Harry shared a room with twin beds
and damp, musty sheets. Dudley snored but Harry stayed awake, sitting on
the windowsill, staring down at the lights of passing cars and
wondering....

They ate stale cornflakes and cold tinned tomatoes on toast for
breakfast the next day. They had just finished when the owner of the
hotel came over to their table.

"\'Scuse me, but is one of you Mr. H. Potter? Only I got about an \'undred
of these at the front desk."

She held up a letter so they could read the green ink address:

Mr. H. Potter

Room 17

Railview Hotel

Cokeworth

Harry made a grab for the letter but Uncle Vernon knocked his hand out
of the way. The woman stared.

"I\'ll take them," said Uncle Vernon, standing up quickly and following
her from the dining room.

Wouldn\'t it be better just to go home, dear?" Aunt Petunia suggested
timidly, hours later, but Uncle Vernon didn\'t seem to hear her. Exactly
what he was looking for, none of them knew. He drove them into the
middle of a forest, got out, looked around, shook his head, got back in
the car, and off they went again. The same thing happened in the middle
of a plowed field, halfway across a suspension bridge, and at the top of
a multilevel parking garage.

"Daddy\'s gone mad, hasn\'t he?" Dudley asked Aunt Petunia dully late that
afternoon. Uncle Vernon had parked at the coast, locked them all inside
the car, and disappeared.


<<Second book>>
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

Food:

Paris is a culinary paradise, offering a wide range of delicious and authentic French food.

Croissants & Pain au Chocolat: The Art of the Bakery: Start your day with a classic French pastry from a boulangerie artisanale (artisanal bakery). Look for bakeries that make their pastries fresh daily using traditional methods.

Macarons: A Sweet Indulgence: Indulge in these colorful and delicate almond meringue cookies. Visit Ladurée and Pierre Hermé for the classic experience, but also explore smaller, independent patisseries for unique flavors.

Steak Frites: A Bistro Classic: Enjoy a quintessential French steak frites at a traditional bistro.

French Onion Soup: A Culinary Comfort: Warm up with a rich and flavorful soupe à l'oignon gratinée (French onion soup) topped with melted cheese.

Local Markets: A Feast for the Senses: Explore local markets like Marché des Enfants Rouges and Marché Bastille for cheese, wine, bread, and other local delicacies.

Bistros: Authentic French Cuisine: Enjoy a traditional French dinner at a bistro. Look for menus that offer "plat du jour" (dish of the day) for a taste of authentic French cuisine.

(Days 4-6: London, England - History and Modernity: Beyond the Landmarks – Extended Edition and Days 7-9: Rome, Italy - Ancient Wonders: Experiencing the Eternal City – Extended Edition will follow the same pattern as above, with detailed elaborations on Accommodation, Activities, and Food, including specific recommendations and insights. Due to space constraints, the full expansion of those sections is not included here. They would continue the same level of detail as the Paris section.)

Day 10-11: Travel Days/Flex Days: Designing Your Own Adventure

These strategically placed "flex days" are crucial to ensuring that your European adventure remains a personal and enriching experience. The pace of the first nine days is deliberately full, immersing you in the highlights of Paris, London, and Rome. However, individual interests and energy levels vary. These two days provide the space to breathe, relax, and pursue activities that truly resonate with you.

Option 1: Rest and Relaxation: Recharge Your Batteries: After a whirlwind tour of three major cities, you might simply need to relax and recharge. This is perfectly acceptable! Use these days to catch up on sleep, relax at your hotel, read a book in a local park, or simply wander around the city without a strict itinerary. Sometimes the best travel experiences come from unplanned moments of serendipity. Find a quiet cafe and people-watch, visit a local spa for a massage, or take a leisurely bike ride through a scenic neighborhood.

Option 2: Day Trip: Expanding Your Horizons: If you're feeling adventurous, consider taking a day trip from one of the cities you've visited. Day trips offer the opportunity to explore a different region, experience a different culture, or visit a specific attraction that interests you.

From Paris: Consider a day trip to the Champagne region, where you can tour vineyards, sample sparkling wine, and learn about the champagne-making process. Alternatively, visit the charming medieval town of Chartres, famous for its magnificent cathedral.

From London: Explore the Cotswolds, a picturesque region of rolling hills, charming villages, and historic manor houses. Another option is to visit Oxford, home to one of the world's oldest and most prestigious universities.

From Rome: A day trip to Pompeii is possible via train, allowing you to explore the remarkably preserved ruins of this ancient Roman city destroyed by the eruption of Mount Vesuvius. A very long day trip to Florence would also be an option, though a longer stay would be recommended to truly experience the city.

Option 3: Revisit Highlights: A Deeper Immersion: If you found a particular sight, museum, or experience that you especially enjoyed, use these days to revisit it for a deeper immersion. Perhaps you want to spend more time in the Louvre, explore a different neighborhood in London, or return to the Roman Forum to wander among the ruins at your own pace.

Day 12-14: Return Home: Reflecting on Your European Odyssey

As your 14-day European adventure draws to a close, it's time to prepare for your departure and reflect on the incredible experiences and memories you've made.

Souvenirs: Purchase souvenirs to remember your trip. Consider buying unique and locally made items that reflect the culture and history of the cities you've visited. Avoid mass-produced tourist trinkets and instead look for handcrafted goods, artwork, or local delicacies.

Sharing Your Stories: Share your stories with friends and family. Relive your favorite moments, show them your photos, and inspire them to plan their own European adventure.

Reflection: Take some time to reflect on the transformative experiences, unforgettable moments, and lasting memories created throughout this European adventure. Consider keeping a journal to record your thoughts and feelings.

Planning Future Trips: Start planning your next European adventure! This 14-day itinerary is just a starting point. There's so much more to explore and discover in Europe.

Budget Considerations:

This itinerary can be adapted to different budgets. Careful planning and smart choices can help you experience the magic of Paris, London, and Rome without breaking the bank.

Accommodation: Hostels and Airbnb offer budget-friendly options. Staying slightly outside the city center can also save money. Consider staying in guesthouses or budget hotels for a more affordable option.

Transportation: Utilize public transport to save money. Walking is a great way to explore the cities. Book train and flight tickets in advance. Look for discounts and promotions on public transport passes.

Food: Eat at local markets and smaller restaurants to save money. Pack snacks and drinks to avoid tourist traps. Opt for street food or picnic lunches instead of expensive restaurants.

Attractions: Purchase a city pass for discounted entry to attractions. Take advantage of free museum days or visit free attractions like parks and churches.

Tips for a Smooth Trip:

Book Well in Advance: Secure accommodation, transportation (especially Eurostar), and popular attractions in advance, especially during peak season.

Pack Light: Pack light to avoid lugging heavy suitcases. Choose versatile clothing items that can be mixed and matched.

Learn Basic Phrases: Learning basic phrases in French, Italian, and English will be helpful.

Stay Connected: Purchase a local SIM card or use a travel eSIM.

Be Aware of Your Surroundings: Be mindful of your belongings and be aware of potential scams.

Wear Comfortable Shoes: Wear comfortable shoes, as you'll be doing a lot of walking.

Embrace the Culture: Be open to new experiences and embrace the local culture. Try new foods, learn about local customs, and interact with the locals.

City-Specific Considerations:

London: London is spread out, so public transport (the Tube) is essential. Consider an Oyster card or contactless payment.

Rome & Paris: These cities are mostly walkable, but utilize the Metro for longer distances.

This 14-day Classic European Capitals Adventure is a springboard for exploration. Customize it according to your interests and budget. With meticulous planning and an adventurous spirit, you're guaranteed an unforgettable journey through some of the world's most beautiful and culturally rich cities. Savor the magic of Paris, the grandeur of London, and the timeless beauty of Rome! Bon voyage! Buongiorno! Have a good trip!


Exploring the Iberian Peninsula: A 14-Day Immersive Journey Through Spain and Portugal (7000 Words)
This expansive 14-day itinerary delves deep into the heart of the Iberian Peninsula, promising an enriching and unforgettable exploration of Spain and Portugal. More than just a surface-level tour, this plan is designed to immerse you in the diverse cultures, tantalizing cuisines, and breathtaking landscapes that define this unique corner of Europe. Prepare for a sun-kissed adventure that blends historical immersion, artistic appreciation, culinary delights, and moments of pure relaxation. From the architectural fantasies of Antoni Gaudí in Barcelona to the soulful strains of Fado music in Lisbon, and the fortified flavors of Port wine in Porto, this journey offers a truly transformative travel experience.

Duration: 14 Days / 13 Nights

Theme: Culture, Cuisine, Beaches, History, Architecture, Music, Wine

Destinations: Barcelona (Spain), Seville (Spain), Lisbon (Portugal), Porto (Portugal)

Plan at a Glance:

Days 1-3: Barcelona, Spain – Unveiling the Catalan Flair: A deep dive into Gaudí's architectural legacy, exploring vibrant markets, savoring Catalan cuisine, and relaxing on Mediterranean beaches. We'll uncover hidden gems and delve into the city's artistic soul.

Days 4-6: Seville, Spain – Experiencing the Andalusian Charm: Immersing ourselves in the historical grandeur of Seville, exploring its magnificent cathedral and royal palace, witnessing the fiery passion of flamenco, indulging in authentic tapas, and considering a captivating day trip to Córdoba.

Days 7-9: Lisbon, Portugal – Ascending the Hills, Unearthing History, and Embracing Fado: Discovering the charm of Lisbon's historic districts, riding the iconic Tram 28, exploring the architectural marvels of Belém, escaping to the fairytale landscapes of Sintra, and surrendering to the emotive power of Fado music.

Days 10-11: Porto, Portugal – Savoring Port Wine and Discovering its Charm: Exploring the historic Ribeira district, marveling at the Dom Luís I Bridge, embarking on a journey into the world of Port wine with a cellar tour, immersing ourselves in literary history at Livraria Lello, and enjoying a scenic cruise on the Douro River.

Days 12-13: Travel Days/Flex Days – Strategic days designed for relaxation, independent exploration, revisiting cherished sites, or accommodating unforeseen circumstances, allowing for a personalized and flexible travel experience.

Day 14: Departure – Reflecting on the remarkable journey through the Iberian Peninsula as you prepare for your departure, carrying with you unforgettable memories and a newfound appreciation for the region's unique character.

Detailed Itinerary:

Day 1-3: Barcelona, Spain - Catalan Flair: Beyond the Surface

Barcelona, a city where art and architecture intertwine, where the Mediterranean sun kisses golden beaches, and where Catalan culture thrives, deserves a thorough exploration. This expanded itinerary digs deeper than the typical tourist trail.

Accommodation:

Gothic Quarter (Barri Gòtic): Beyond its central location and medieval architecture, the Gothic Quarter holds a wealth of secrets. Explore its hidden courtyards, discover Roman ruins beneath the city streets, and uncover the stories behind its ancient buildings. Consider a guided walking tour to uncover its hidden history.

El Born: Adjacent to the Gothic Quarter, El Born pulsates with contemporary creativity. Beyond its boutiques and galleries, explore its independent workshops, artisan studios, and vibrant nightlife scene. Discover the Picasso Museum, showcasing the artist's early works and connection to Barcelona.

Considerations: Research accommodation options based on your preferred atmosphere. The Gothic Quarter offers historical charm, while El Born provides a trendier vibe. Consider the proximity to metro stations for seamless exploration. Air conditioning is a must during the hot summer months.

Activities:

Arrival & Alternative Ramblas Exploration: Skip the tourist hordes on Las Ramblas and explore the less crowded but equally charming La Rambla de Catalunya. This more upscale boulevard offers wider sidewalks, designer boutiques, and beautiful modernist buildings.

Sagrada Família: An Architectural Revelation: Pre-booked tickets are paramount. Allocate ample time (at least 3 hours) to truly appreciate the Sagrada Família. Consider booking a guided tour that delves into Gaudí's symbolism, construction techniques, and the ongoing architectural challenges. Explore the museum below the basilica to understand the history and future plans. Climb one of the towers for panoramic views, but be aware of the potential for long wait times.

Park Güell: A Whimsical Escape: Beyond the iconic mosaic benches, explore the quieter corners of Park Güell. Discover the Casa Museu Gaudí, where the architect lived, and learn about his personal life and creative process. Attend a sunset session in the park to experience the magical atmosphere as the city lights begin to twinkle.

Casa Batlló & Casa Milà (La Pedrera): Modernist Masterpieces: Instead of just a quick visit, dedicate a half-day to exploring both Casa Batlló and Casa Milà. At Casa Batlló, admire the organic forms, vibrant colors, and intricate details that evoke the underwater world. At Casa Milà, explore the rooftop terrace with its surreal chimney sculptures and enjoy panoramic city views. Consider an evening visit to Casa Batlló for a magical light and sound show.

Boqueria Market: A Culinary Journey: Beyond the visual spectacle, engage with the vendors, sample local delicacies, and learn about Catalan culinary traditions. Try jamón ibérico, fresh seafood, local cheeses, and seasonal fruits. Arrive early in the morning to avoid the crowds and experience the market at its most vibrant. Consider a cooking class that utilizes ingredients from the Boqueria Market.

Barceloneta Beach: Beyond the Sand: Explore the Barceloneta neighborhood, a former fishing village, and sample fresh seafood at one of the beachfront restaurants. Take a stroll along the promenade, rent a bike, or try stand-up paddleboarding. Escape the crowds by venturing to one of the less-known beaches further north or south of Barceloneta.

Montjuïc Hill: A Panoramic Perspective: Allocate an entire day to exploring Montjuïc Hill. Visit the Montjuïc Castle for panoramic views and historical insights. Explore the Joan Miró Foundation, showcasing the works of the renowned Catalan artist. Visit the Olympic Park, a legacy of the 1992 Barcelona Olympics. Attend the Magic Fountain of Montjuïc show at night, a spectacular display of water, light, and music. Take the cable car up the hill for stunning views of the city.

Food:

Tapas Exploration: Venture beyond the tourist traps and explore the tapas bars in the El Born and Gràcia neighborhoods. Try regional specialties like pa amb tomàquet (bread rubbed with tomato), escalivada (grilled vegetables), and bombas (potato croquettes with spicy sauce).

Paella Variations: While paella is a must-try, explore the different variations, such as arròs negre (black rice paella with squid ink) and fideuà (paella made with noodles instead of rice).

Cava Tasting: Sample local Cava, a sparkling wine from the Penedès region, at a Cava bar or vineyard.

Crema Catalana Indulgence: Try different variations of crema catalana, such as those flavored with citrus or cinnamon.

Hidden Bodegas: Discover hidden bodegas in the Gothic Quarter and sample local wines.

Day 4-6: Seville, Spain - Andalusian Charm: Into the Heart of Andalusia

Seville, the vibrant capital of Andalusia, is a city that captivates with its Moorish heritage, passionate flamenco rhythms, and sun-drenched streets. This expanded itinerary takes you beyond the typical tourist sights and into the heart of Andalusian culture.

Transportation:

High-Speed Train (AVE): Book your AVE train tickets well in advance to secure the best prices and preferred seating. Consider purchasing a Renfe Spain Pass if you plan on traveling extensively by train in Spain.

Accommodation:

Santa Cruz Neighborhood (Old Jewish Quarter): A Deeper Dive: While beautiful, Santa Cruz can be crowded. Consider staying in the neighboring El Arenal district, which offers a more local atmosphere and is still within walking distance of the major attractions.

Considerations: Look for accommodation with a rooftop terrace to enjoy panoramic city views. Be aware that Santa Cruz can be noisy at night.

Activities:

Seville Cathedral & Giralda: Architectural Grandeur: Allocate ample time (at least 3 hours) to explore the Seville Cathedral and Giralda. Climb to the top of the Giralda for breathtaking views of the city. Explore the cathedral's chapels, admire its artwork, and learn about its history.

Alcázar of Seville: A Royal Retreat: Explore the Alcázar's hidden corners, admire its intricate tilework, and wander through its lush gardens. Learn about the Alcázar's history as a royal residence and its influence on Andalusian architecture. Consider a guided tour to gain a deeper understanding of its historical significance.

Plaza de España: A Symbol of Spain: Take a horse-drawn carriage ride around Plaza de España for a unique perspective. Rent a rowboat and explore the canal. Visit the museums and cultural centers located within the plaza. Attend a performance or event at the plaza.

Flamenco: A Cultural Immersion: Choose a flamenco show that features authentic performers and traditional music. Learn about the history and different styles of flamenco. Consider taking a flamenco dance lesson. Dress up in traditional flamenco attire.

Cooking Class: A Culinary Journey: Learn to make traditional Andalusian dishes, such as gazpacho, salmorejo, paella, and tapas. Visit a local market to purchase fresh ingredients. Sample local wines and sherries. Enjoy the fruits of your labor with a delicious meal.

Guadalquivir River: A Scenic Cruise: Take a river cruise on the Guadalquivir River and admire the views of the city. Learn about the river's history and its importance to Seville. Enjoy a sunset cruise for a romantic experience.

Tapas Tour: A Culinary Adventure: Explore the tapas bars in the Triana neighborhood, known for its authentic tapas and lively atmosphere. Try regional specialties like pescaíto frito (fried fish) and espinacas con garbanzos (spinach with chickpeas).

Córdoba (Day Trip): A Mesmerizing Blend: Take a day trip to Córdoba to see the Mezquita-Cathedral, a stunning example of Moorish architecture. Explore the Jewish Quarter, wander through its narrow streets, and visit its synagogues. Visit the Alcázar de los Reyes Cristianos, a former royal residence.

Food:

Tapas Discovery: Explore the diverse world of tapas in Seville. Venture beyond the typical tourist fare and try regional specialties. Ask the locals for recommendations.

Sherry Tasting: Sample local sherry wines at a sherry bar. Learn about the different types of sherry and their production methods.

Orange Grove Visit: Visit an orange grove and learn about the cultivation and harvesting of Seville oranges. Sample fresh orange juice and other orange-based products.

Explore Triana Market: Visit the Triana Market and experience local Andalusian food.

Day 7-9: Lisbon, Portugal - Hills, History, and Heart: Unveiling Lisbon's Soul

Lisbon, Portugal's captivating capital, is a city of contrasts, where ancient history blends seamlessly with modern vibrancy. This enhanced itinerary delves deeper into Lisbon's soul, exploring its hidden corners, savoring its culinary delights, and immersing ourselves in its rich cultural heritage.

Transportation:

Lisbon Card: Consider purchasing a Lisbon Card for unlimited access to public transport and free entry to many attractions.

Accommodation:

Alfama District: Immersing in Authenticity: Beyond its Fado houses and narrow streets, Alfama offers a glimpse into Lisbon's traditional way of life. Seek out smaller, family-run guesthouses for a more authentic experience. Be prepared for hills and cobblestone streets.

Baixa District: Central Convenience: While Baixa is central, it can be touristy. Consider staying in the neighboring Chiado district, which offers a more upscale atmosphere and is still within walking distance of the major attractions.

Considerations: Consider accommodation options with a miradouro (viewpoint) for stunning city views.

Activities:

Alfama: Beyond the Postcard: Get intentionally lost in the Alfama, discovering hidden squares, local shops, and traditional cafes. Visit the Lisbon Cathedral, the oldest church in the city. Explore the Roman Theatre Museum, showcasing the ruins of a Roman theatre.

São Jorge Castle: History and Panoramic Views: Explore the castle's ramparts, admire the views, and learn about its history as a royal residence. Attend a performance or event at the castle.

Tram 28: A Classic Lisbon Experience: Ride Tram 28 early in the morning to avoid the crowds. Sit by the window to enjoy the best views. Be aware of pickpockets on the tram.

Jerónimos Monastery: A Maritime Legacy: Take your time exploring the Jerónimos Monastery, admiring its intricate carvings and learning about its connection to the Age of Discovery.

Belém Tower: A Guardian of Lisbon: Explore Belém Tower's different levels, admire its architecture, and learn about its role in protecting Lisbon.

Pastéis de Belém: A Sweet Tradition: Enjoy a Pastel de Belém with a sprinkle of cinnamon and powdered sugar. Try other traditional Portuguese pastries.

Sintra: A Fairytale Escape: Take a full day trip to Sintra, exploring Pena Palace, Quinta da Regaleira, and the Moorish Castle. Wear comfortable shoes for walking.

Fado: An Evening of Soulful Music: Research different Fado houses and choose one that offers an authentic experience. Make a reservation in advance. Dress respectfully. Listen attentively to the music.

Food:

Seafood Extravaganza: Explore Lisbon's seafood restaurants and try dishes like cataplana (seafood stew), grilled sardines, and bacalhau à brás (shredded salt cod with eggs and potatoes).

Bacalhau (Salt Cod): A Portuguese Staple: Try different variations of bacalhau, as the Portuguese claim to have 365 ways to cook it.

Vinho Verde (Green Wine): A Refreshing Delight: Sample different types of vinho verde from the Minho region.

Pastéis de Nata Workshop: Take a class and learn how to make Lisbon's famous custard tarts.

Day 10-11: Porto, Portugal - Port Wine and Riverside Charm: A Northern Gem

Porto, Portugal's second city, is a captivating destination that enchants visitors with its historic Ribeira district, iconic Dom Luís I Bridge, and world-renowned Port wine cellars. This enhanced itinerary explores Porto's unique charm, delving into its culinary delights and cultural treasures.

Transportation:

Andante Tour Card: Consider purchasing an Andante Tour card for unlimited travel on Porto's public transport.

Accommodation:

Ribeira District: Immersing in History and Atmosphere: Choose accommodation with views of the Douro River and Luís I Bridge. Be prepared for steep streets and lively nightlife.

Vila Nova de Gaia: Port Wine Haven: Stay close to the Port wine cellars for easy access to tours and tastings. Enjoy quieter nights than in the Ribeira district.

Considerations: Booking in advance is highly recommended, especially during peak season.

Activities:

Ribeira: A UNESCO World Heritage Site: Explore the Ribeira district on foot, admiring its colorful buildings, narrow streets, and historic charm. Visit the Palácio da Bolsa, a stunning stock exchange palace. Explore the São Francisco Church, known for its opulent gold-leaf interior.

Dom Luís I Bridge: An Engineering Marvel: Walk across the Dom Luís I Bridge and admire the panoramic views of Porto and Vila Nova de Gaia. Take photos from the bridge at sunset for a magical experience.

Port Wine Cellars: A Journey Through Flavor: Take a guided tour of a Port wine cellar and learn about the production process. Sample different types of Port wine, from tawny to ruby. Learn about the history of Port wine and its connection to Porto.

Livraria Lello: A Literary Gem: Admire the bookstore's stunning architecture, including its spiral staircase and stained-glass ceiling.

Douro River Cruise: A Scenic Voyage: Take a boat trip on the Douro River and admire the views of Porto and Vila Nova de Gaia. Learn about the history of the river and its importance to the region. Enjoy a sunset cruise for a romantic experience.

Clérigos Church and Tower: A Panoramic Climb: Climb to the top of the Clérigos Tower for panoramic views of Porto. Admire the church's baroque architecture.

Francesinha: A Local Delicacy: Sample a Francesinha sandwich at a local restaurant. Try different variations of the sandwich.

Food:

Francesinha Exploration: Try Francesinha sandwich at various restaurants and discuss your opinion with locals.

Seafood Delight: Try various seafood options, specially fresh fishes.

Explore Local bakeries: Enjoy a Portuguese Custard Tart on the go.

Port Wine and Chocolate Pairing: Indulge in a Port wine and chocolate pairing experience.

Day 12-13: Flex Days: Your Time, Your Choice

These strategically placed flex days empower you to personalize your itinerary based on your preferences and discoveries.

Options:

Relaxation and Reflection: Dedicate a day to relaxation, allowing time to process your experiences and recharge for the journey ahead. Enjoy a spa treatment, relax by a pool, or simply unwind in your hotel room.

Independent Exploration: Venture off the beaten path and discover hidden gems in any of the cities you've visited.

Revisit Cherished Sites: Return to your favorite attractions for a more in-depth exploration or simply to soak in the atmosphere.

Unforeseen Circumstances: Utilize these days to accommodate unforeseen travel delays or personal preferences.

Day 14: Departure

Prepare for your departure from Porto Airport (OPO), allowing ample time for airport procedures.

Reflect on the incredible journey through the Iberian Peninsula, cherishing the memories and insights gained.

Purchase any last-minute souvenirs to commemorate your adventure.

This comprehensive 14-day itinerary offers a deeper and more immersive exploration of the Iberian Peninsula, promising a truly transformative travel experience. Embrace the rich cultures, savor the delectable cuisines, and discover the hidden gems that make Spain and Portugal so captivating. Enjoy your sun-kissed adventure!

Summarization:""",
        """There are two books, named <<First book>> and <<Second book>>, Pls do Summary and compare the content for these two books.
You must tell the difference between two books.
you will focus on identifying potential key themes, probable main plot points or central arguments, 
and the overall tone of each book. You will approach this task assuming a broad readership, 
creating summaries suitable for someone unfamiliar with the works. You will focusing on the core ideas and potential takeaways.
You'll do my best to glean the heart of each book and present it in a clear and helpful manner. 


<<First book>>
Harry Potter and the Sorcerer\'s Stone


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
half-moon glasses. "It would be enough to turn any boy\'s head. Famous
before he can walk and talk! Famous for something he won\'t even
remember! CarA you see how much better off he\'ll be, growing up away
from all that until he\'s ready to take it?"

Professor McGonagall opened her mouth, changed her mind, swallowed, and
then said, "Yes -- yes, you\'re right, of course. But how is the boy
getting here, Dumbledore?" She eyed his cloak suddenly as though she
thought he might be hiding Harry underneath it.

"Hagrid\'s bringing him."

"You think it -- wise -- to trust Hagrid with something as important as
this?"

I would trust Hagrid with my life," said Dumbledore.

"I\'m not saying his heart isn\'t in the right place," said Professor
McGonagall grudgingly, "but you can\'t pretend he\'s not careless. He does
tend to -- what was that?"

A low rumbling sound had broken the silence around them. It grew
steadily louder as they looked up and down the street for some sign of a
headlight; it swelled to a roar as they both looked up at the sky -- and
a huge motorcycle fell out of the air and landed on the road in front of
them.

If the motorcycle was huge, it was nothing to the man sitting astride
it. He was almost twice as tall as a normal man and at least five times
as wide. He looked simply too big to be allowed, and so wild - long
tangles of bushy black hair and beard hid most of his face, he had hands
the size of trash can lids, and his feet in their leather boots were
like baby dolphins. In his vast, muscular arms he was holding a bundle
of blankets.

"Hagrid," said Dumbledore, sounding relieved. "At last. And where did
you get that motorcycle?"

"Borrowed it, Professor Dumbledore, sit," said the giant, climbing
carefully off the motorcycle as he spoke. "Young Sirius Black lent it to
me. I\'ve got him, sir."

"No problems, were there?"

"No, sir -- house was almost destroyed, but I got him out all right
before the Muggles started swarmin\' around. He fell asleep as we was
flyin\' over Bristol."

Dumbledore and Professor McGonagall bent forward over the bundle of
blankets. Inside, just visible, was a baby boy, fast asleep. Under a
tuft of jet-black hair over his forehead they could see a curiously
shaped cut, like a bolt of lightning.

"Is that where -?" whispered Professor McGonagall.

"Yes," said Dumbledore. "He\'ll have that scar forever."

"Couldn\'t you do something about it, Dumbledore?"

"Even if I could, I wouldn\'t. Scars can come in handy. I have one myself
above my left knee that is a perfect map of the London Underground. Well
-- give him here, Hagrid -- we\'d better get this over with."

Dumbledore took Harry in his arms and turned toward the Dursleys\' house.

"Could I -- could I say good-bye to him, sir?" asked Hagrid. He bent his
great, shaggy head over Harry and gave him what must have been a very
scratchy, whiskery kiss. Then, suddenly, Hagrid let out a howl like a
wounded dog.

"Shhh!" hissed Professor McGonagall, "you\'ll wake the Muggles!"

"S-s-sorry," sobbed Hagrid, taking out a large, spotted handkerchief and
burying his face in it. "But I c-c-can\'t stand it -- Lily an\' James dead
-- an\' poor little Harry off ter live with Muggles -"

"Yes, yes, it\'s all very sad, but get a grip on yourself, Hagrid, or
we\'ll be found," Professor McGonagall whispered, patting Hagrid gingerly
on the arm as Dumbledore stepped over the low garden wall and walked to
the front door. He laid Harry gently on the doorstep, took a letter out
of his cloak, tucked it inside Harry\'s blankets, and then came back to
the other two. For a full minute the three of them stood and looked at
the little bundle; Hagrid\'s shoulders shook, Professor McGonagall
blinked furiously, and the twinkling light that usually shone from
Dumbledore\'s eyes seemed to have gone out.

"Well," said Dumbledore finally, "that\'s that. We\'ve no business staying
here. We may as well go and join the celebrations."

"Yeah," said Hagrid in a very muffled voice, "I\'ll be takin\' Sirius his
bike back. G\'night, Professor McGonagall -- Professor Dumbledore, sir."

Wiping his streaming eyes on his jacket sleeve, Hagrid swung himself
onto the motorcycle and kicked the engine into life; with a roar it rose
into the air and off into the night.

"I shall see you soon, I expect, Professor McGonagall," said Dumbledore,
nodding to her. Professor McGonagall blew her nose in reply.

Dumbledore turned and walked back down the street. On the corner he
stopped and took out the silver Put-Outer. He clicked it once, and
twelve balls of light sped back to their street lamps so that Privet
Drive glowed suddenly orange and he could make out a tabby cat slinking
around the corner at the other end of the street. He could just see the
bundle of blankets on the step of number four.

"Good luck, Harry," he murmured. He turned on his heel and with a swish
of his cloak, he was gone.

A breeze ruffled the neat hedges of Privet Drive, which lay silent and
tidy under the inky sky, the very last place you would expect
astonishing things to happen. Harry Potter rolled over inside his
blankets without waking up. One small hand closed on the letter beside
him and he slept on, not knowing he was special, not knowing he was
famous, not knowing he would be woken in a few hours\' time by Mrs.
Dursley\'s scream as she opened the front door to put out the milk
bottles, nor that he would spend the next few weeks being prodded and
pinched by his cousin Dudley... He couldn\'t know that at this very
moment, people meeting in secret all over the country were holding up
their glasses and saying in hushed voices: "To Harry Potter -- the boy
who lived!"

"S-s-sorry," sobbed Hagrid, taking out a large, spotted handkerchief and
burying his face in it. "But I c-c-can\'t stand it -- Lily an\' James dead
-- an\' poor little Harry off ter live with Muggles -"

"Yes, yes, it\'s all very sad, but get a grip on yourself, Hagrid, or
we\'ll be found," Professor McGonagall whispered, patting Hagrid gingerly
on the arm as Dumbledore stepped over the low garden wall and walked to
the front door. He laid Harry gently on the doorstep, took a letter out
of his cloak, tucked it inside Harry\'s blankets, and then came back to
the other two. For a full minute the three of them stood and looked at
the little bundle; Hagrid\'s shoulders shook, Professor McGonagall
blinked furiously, and the twinkling light that usually shone from
Dumbledore\'s eyes seemed to have gone out.

"Well," said Dumbledore finally, "that\'s that. We\'ve no business staying
here. We may as well go and join the celebrations."

"Yeah," said Hagrid in a very muffled voice, "I\'ll be takin\' Sirius his
bike back. G\'night, Professor McGonagall -- Professor Dumbledore, sir."

Wiping his streaming eyes on his jacket sleeve, Hagrid swung himself
onto the motorcycle and kicked the engine into life; with a roar it rose
into the air and off into the night.

"I shall see you soon, I expect, Professor McGonagall," said Dumbledore,
nodding to her. Professor McGonagall blew her nose in reply.

Dumbledore turned and walked back down the street. On the corner he
stopped and took out the silver Put-Outer. He clicked it once, and
twelve balls of light sped back to their street lamps so that Privet
Drive glowed suddenly orange and he could make out a tabby cat slinking
around the corner at the other end of the street. He could just see the
bundle of blankets on the step of number four.

"Good luck, Harry," he murmured. He turned on his heel and with a swish
of his cloak, he was gone.



CHAPTER TWO

THE VANISHING GLASS

Nearly ten years had passed since the Dursleys had woken up to find
their nephew on the front step, but Privet Drive had hardly changed at
all. The sun rose on the same tidy front gardens and lit up the brass
number four on the Dursleys\' front door; it crept into their living
room, which was almost exactly the same as it had been on the night when
Mr. Dursley had seen that fateful news report about the owls. Only the
photographs on the mantelpiece really showed how much time had passed.
Ten years ago, there had been lots of pictures of what looked like a
large pink beach ball wearing different-colored bonnets -- but Dudley
Dursley was no longer a baby, and now the photographs showed a large
blond boy riding his first bicycle, on a carousel at the fair, playing a
computer game with his father, being hugged and kissed by his mother.
The room held no sign at all that another boy lived in the house, too.

Yet Harry Potter was still there, asleep at the moment, but not for
long. His Aunt Petunia was awake and it was her shrill voice that made
the first noise of the day.

"Up! Get up! Now!"

Harry woke with a start. His aunt rapped on the door again.

"Up!" she screeched. Harry heard her walking toward the kitchen and then
the sound of the frying pan being put on the stove. He rolled onto his
back and tried to remember the dream he had been having. It had been a
good one. There had been a flying motorcycle in it. He had a funny
feeling he\'d had the same dream before.

His aunt was back outside the door.

"Are you up yet?" she demanded.

"Nearly," said Harry.

"Well, get a move on, I want you to look after the bacon. And don\'t you
dare let it burn, I want everything perfect on Duddy\'s birthday."

Harry groaned.

"What did you say?" his aunt snapped through the door.

"Nothing, nothing..."

Dudley\'s birthday -- how could he have forgotten? Harry got slowly out
of bed and started looking for socks. He found a pair under his bed and,
after pulling a spider off one of them, put them on. Harry was used to
spiders, because the cupboard under the stairs was full of them, and
that was where he slept.

When he was dressed he went down the hall into the kitchen. The table
was almost hidden beneath all Dudley\'s birthday presents. It looked as
though Dudley had gotten the new computer he wanted, not to mention the
second television and the racing bike. Exactly why Dudley wanted a
racing bike was a mystery to Harry, as Dudley was very fat and hated
exercise -- unless of course it involved punching somebody. Dudley\'s
favorite punching bag was Harry, but he couldn\'t often catch him. Harry
didn\'t look it, but he was very fast.

Perhaps it had something to do with living in a dark cupboard, but Harry
had always been small and skinny for his age. He looked even smaller and
skinnier than he really was because all he had to wear were old clothes
of Dudley\'s, and Dudley was about four times bigger than he was. Harry
had a thin face, knobbly knees, black hair, and bright green eyes. He
wore round glasses held together with a lot of Scotch tape because of
all the times Dudley had punched him on the nose. The only thing Harry
liked about his own appearance was a very thin scar on his forehead that
was shaped like a bolt of lightning. He had had it as long as he could
remember, and the first question he could ever remember asking his Aunt
Petunia was how he had gotten it.

"In the car crash when your parents died," she had said. "And don\'t ask
questions."

Don\'t ask questions -- that was the first rule for a quiet life with the
Dursleys.

Uncle Vernon entered the kitchen as Harry was turning over the bacon.

"Comb your hair!" he barked, by way of a morning greeting.

About once a week, Uncle Vernon looked over the top of his newspaper and
shouted that Harry needed a haircut. Harry must have had more haircuts
than the rest of the boys in his class put

together, but it made no difference, his hair simply grew that way --
all over the place.

Harry was frying eggs by the time Dudley arrived in the kitchen with his
mother. Dudley looked a lot like Uncle Vernon. He had a large pink face,
not much neck, small, watery blue eyes, and thick blond hair that lay
smoothly on his thick, fat head. Aunt Petunia often said that Dudley
looked like a baby angel -- Harry often said that Dudley looked like a
pig in a wig.

Harry put the plates of egg and bacon on the table, which was difficult
as there wasn\'t much room. Dudley, meanwhile, was counting his presents.
His face fell.

"Thirty-six," he said, looking up at his mother and father. "That\'s two
less than last year."

"Darling, you haven\'t counted Auntie Marge\'s present, see, it\'s here
under this big one from Mommy and Daddy."

"All right, thirty-seven then," said Dudley, going red in the face.
Harry, who could see a huge Dudley tantrum coming on, began wolfing down
his bacon as fast as possible in case Dudley turned the table over.

Aunt Petunia obviously scented danger, too, because she said quickly,
"And we\'ll buy you another two presents while we\'re out today. How\'s
that, popkin? Two more presents. Is that all right\'\'

Dudley thought for a moment. It looked like hard work. Finally he said
slowly, "So I\'ll have thirty ... thirty..."

"Thirty-nine, sweetums," said Aunt Petunia.

"Oh." Dudley sat down heavily and grabbed the nearest parcel. "All right
then."

Uncle Vernon chuckled. "Little tyke wants his money\'s worth, just like
his father. \'Atta boy, Dudley!" He ruffled Dudley\'s hair.

At that moment the telephone rang and Aunt Petunia went to answer it
while Harry and Uncle Vernon watched Dudley unwrap the racing bike, a
video camera, a remote control airplane, sixteen new computer games, and
a VCR. He was ripping the paper off a gold wristwatch when Aunt Petunia
came back from the telephone looking both angry and worried.

"Bad news, Vernon," she said. "Mrs. Figg\'s broken her leg. She can\'t
take him." She jerked her head in Harry\'s direction.

Dudley\'s mouth fell open in horror, but Harry\'s heart gave a leap. Every
year on Dudley\'s birthday, his parents took him and a friend out for the
day, to adventure parks, hamburger restaurants, or the movies. Every
year, Harry was left behind with Mrs. Figg, a mad old lady who lived two
streets away. Harry hated it there. The whole house smelled of cabbage
and Mrs. Figg made him look at photographs of all the cats she\'d ever
owned.

"Now what?" said Aunt Petunia, looking furiously at Harry as though he\'d
planned this. Harry knew he ought to feel sorry that Mrs. Figg had
broken her leg, but it wasn\'t easy when he reminded himself it would be
a whole year before he had to look at Tibbles, Snowy, Mr. Paws, and
Tufty again.

"We could phone Marge," Uncle Vernon suggested.

"Don\'t be silly, Vernon, she hates the boy."

The Dursleys often spoke about Harry like this, as though he wasn\'t
there -- or rather, as though he was something very nasty that couldn\'t
understand them, like a slug.

"What about what\'s-her-name, your friend -- Yvonne?"

"On vacation in Majorca," snapped Aunt Petunia.

"You could just leave me here," Harry put in hopefully (he\'d be able to
watch what he wanted on television for a change and maybe even have a go
on Dudley\'s computer).

Aunt Petunia looked as though she\'d just swallowed a lemon.

"And come back and find the house in ruins?" she snarled.

"I won\'t blow up the house," said Harry, but they weren\'t listening.

"I suppose we could take him to the zoo," said Aunt Petunia slowly, "...
and leave him in the car...."

"That car\'s new, he\'s not sitting in it alone...."

Dudley began to cry loudly. In fact, he wasn\'t really crying -- it had
been years since he\'d really cried -- but he knew that if he screwed up
his face and wailed, his mother would give him anything he wanted.

"Dinky Duddydums, don\'t cry, Mummy won\'t let him spoil your special
day!" she cried, flinging her arms around him.

"I... don\'t... want... him... t-t-to come!" Dudley yelled between huge,
pretend sobs. "He always sp- spoils everything!" He shot Harry a nasty
grin through the gap in his mother\'s arms.

Just then, the doorbell rang -- "Oh, good Lord, they\'re here!" said Aunt
Petunia frantically -- and a moment later, Dudley\'s best friend, Piers
Polkiss, walked in with his mother. Piers was a scrawny boy with a face
like a rat. He was usually the one who held people\'s arms behind their
backs while Dudley hit them. Dudley stopped pretending to cry at once.

Half an hour later, Harry, who couldn\'t believe his luck, was sitting in
the back of the Dursleys\' car with Piers and Dudley, on the way to the
zoo for the first time in his life. His aunt and uncle hadn\'t been able
to think of anything else to do with him, but before they\'d left, Uncle
Vernon had taken Harry aside.

"I\'m warning you," he had said, putting his large purple face right up
close to Harry\'s, "I\'m warning you now, boy -- any funny business,
anything at all -- and you\'ll be in that cupboard from now until
Christmas."

"I\'m not going to do anything," said Harry, "honestly..

But Uncle Vernon didn\'t believe him. No one ever did.

The problem was, strange things often happened around Harry and it was
just no good telling the Dursleys he didn\'t make them happen.

Once, Aunt Petunia, tired of Harry coming back from the barbers looking
as though he hadn\'t been at all, had taken a pair of kitchen scissors
and cut his hair so short he was almost bald except for his bangs, which
she left "to hide that horrible scar." Dudley had laughed himself silly
at Harry, who spent a sleepless night imagining school the next day,
where he was already laughed at for his baggy clothes and taped glasses.
Next morning, however, he had gotten up to find his hair exactly as it
had been before Aunt Petunia had sheared it off He had been given a week
in his cupboard for this, even though he had tried to explain that he
couldn\'t explain how it had grown back so quickly.

Another time, Aunt Petunia had been trying to force him into a revolting
old sweater of Dudley\'s (brown with orange puff balls) -- The harder she
tried to pull it over his head, the smaller it seemed to become, until
finally it might have fitted a hand puppet, but certainly wouldn\'t fit
Harry. Aunt Petunia had decided it must have shrunk in the wash and, to
his great relief, Harry wasn\'t punished.

On the other hand, he\'d gotten into terrible trouble for being found on
the roof of the school kitchens. Dudley\'s gang had been chasing him as
usual when, as much to Harry\'s surprise as anyone else\'s, there he was
sitting on the chimney. The Dursleys had received a very angry letter
from Harry\'s headmistress telling them Harry had been climbing school
buildings. But all he\'d tried to do (as he shouted at Uncle Vernon
through the locked door of his cupboard) was jump behind the big trash
cans outside the kitchen doors. Harry supposed that the wind must have
caught him in mid- jump.

But today, nothing was going to go wrong. It was even worth being with
Dudley and Piers to be spending the day somewhere that wasn\'t school,
his cupboard, or Mrs. Figg\'s cabbage-smelling living room.

While he drove, Uncle Vernon complained to Aunt Petunia. He liked to
complain about things: people at work, Harry, the council, Harry, the
bank, and Harry were just a few of his favorite subjects. This morning,
it was motorcycles.

"... roaring along like maniacs, the young hoodlums," he said, as a
motorcycle overtook them.

I had a dream about a motorcycle," said Harry, remembering suddenly. "It
was flying."

Uncle Vernon nearly crashed into the car in front. He turned right
around in his seat and yelled at Harry, his face like a gigantic beet
with a mustache: "MOTORCYCLES DON\'T FLY!"

Dudley and Piers sniggered.

I know they don\'t," said Harry. "It was only a dream."

But he wished he hadn\'t said anything. If there was one thing the
Dursleys hated even more than his asking questions, it was his talking
about anything acting in a way it shouldn\'t, no matter if it was in a
dream or even a cartoon -- they seemed to think he might get dangerous
ideas.

It was a very sunny Saturday and the zoo was crowded with families. The
Dursleys bought Dudley and Piers large chocolate ice creams at the
entrance and then, because the smiling lady in the van had asked Harry
what he wanted before they could hurry him away, they bought him a cheap
lemon ice pop. It wasn\'t bad, either, Harry thought, licking it as they
watched a gorilla scratching its head who looked remarkably like Dudley,
except that it wasn\'t blond.

Harry had the best morning he\'d had in a long time. He was careful to
walk a little way apart from the Dursleys so that Dudley and Piers, who
were starting to get bored with the animals by lunchtime, wouldn\'t fall
back on their favorite hobby of hitting him. They ate in the zoo
restaurant, and when Dudley had a tantrum because his knickerbocker
glory didn\'t have enough ice cream on top, Uncle Vernon bought him
another one and Harry was allowed to finish the first.

Harry felt, afterward, that he should have known it was all too good to
last.

After lunch they went to the reptile house. It was cool and dark in
there, with lit windows all along the walls. Behind the glass, all sorts
of lizards and snakes were crawling and slithering over bits of wood and
stone. Dudley and Piers wanted to see huge, poisonous cobras and thick,
man-crushing pythons. Dudley quickly found the largest snake in the
place. It could have wrapped its body twice around Uncle Vernon\'s car
and crushed it into a trash can -- but at the moment it didn\'t look in
the mood. In fact, it was fast asleep.

Dudley stood with his nose pressed against the glass, staring at the
glistening brown coils.

"Make it move," he whined at his father. Uncle Vernon tapped on the
glass, but the snake didn\'t budge.

"Do it again," Dudley ordered. Uncle Vernon rapped the glass smartly
with his knuckles, but the snake just snoozed on.

"This is boring," Dudley moaned. He shuffled away.

Harry moved in front of the tank and looked intently at the snake. He
wouldn\'t have been surprised if it had died of boredom itself -- no
company except stupid people drumming their fingers on the glass trying
to disturb it all day long. It was worse than having a cupboard as a
bedroom, where the only visitor was Aunt Petunia hammering on the door
to wake you up; at least he got to visit the rest of the house.

The snake suddenly opened its beady eyes. Slowly, very slowly, it raised
its head until its eyes were on a level with Harry\'s.

It winked.

Harry stared. Then he looked quickly around to see if anyone was
watching. They weren\'t. He looked back at the snake and winked, too.

The snake jerked its head toward Uncle Vernon and Dudley, then raised
its eyes to the ceiling. It gave Harry a look that said quite plainly:

"I get that all the time.

"I know," Harry murmured through the glass, though he wasn\'t sure the
snake could hear him. "It must be really annoying."

The snake nodded vigorously.

"Where do you come from, anyway?" Harry asked.

The snake jabbed its tail at a little sign next to the glass. Harry
peered at it.

Boa Constrictor, Brazil.

"Was it nice there?"

The boa constrictor jabbed its tail at the sign again and Harry read on:
This specimen was bred in the zoo. "Oh, I see -- so you\'ve never been to
Brazil?"

As the snake shook its head, a deafening shout behind Harry made both of
them jump.

"DUDLEY! MR. DURSLEY! COME AND LOOK AT THIS SNAKE! YOU WON\'T BELIEVE
WHAT IT\'S DOING!"

Dudley came waddling toward them as fast as he could.

"Out of the way, you," he said, punching Harry in the ribs. Caught by
surprise, Harry fell hard on the concrete floor. What came next happened
so fast no one saw how it happened -- one second, Piers and Dudley were
leaning right up close to the glass, the next, they had leapt back with
howls of horror.

Harry sat up and gasped; the glass front of the boa constrictor\'s tank
had vanished. The great snake was uncoiling itself rapidly, slithering
out onto the floor. People throughout the reptile house screamed and
started running for the exits.

As the snake slid swiftly past him, Harry could have sworn a low,
hissing voice said, "Brazil, here I come.... Thanksss, amigo."

The keeper of the reptile house was in shock.

"But the glass," he kept saying, "where did the glass go?"

The zoo director himself made Aunt Petunia a cup of strong, sweet tea
while he apologized over and over again. Piers and Dudley could only
gibber. As far as Harry had seen, the snake hadn\'t done anything except
snap playfully at their heels as it passed, but by the time they were
all back in Uncle Vernon\'s car, Dudley was telling them how it had
nearly bitten off his leg, while Piers was swearing it had tried to
squeeze him to death. But worst of all, for Harry at least, was Piers
calming down enough to say, "Harry was talking to it, weren\'t you,
Harry?"

Uncle Vernon waited until Piers was safely out of the house before
starting on Harry. He was so angry he could hardly speak. He managed to
say, "Go -- cupboard -- stay -- no meals," before he collapsed into a
chair, and Aunt Petunia had to run and get him a large brandy.

Harry lay in his dark cupboard much later, wishing he had a watch. He
didn\'t know what time it was and he couldn\'t be sure the Dursleys were
asleep yet. Until they were, he couldn\'t risk sneaking to the kitchen
for some food.

He\'d lived with the Dursleys almost ten years, ten miserable years, as
long as he could remember, ever since he\'d been a baby and his parents
had died in that car crash. He couldn\'t remember being in the car when
his parents had died. Sometimes, when he strained his memory during long
hours in his cupboard, he came up with a strange vision: a blinding
flash of green light and a burn- ing pain on his forehead. This, he
supposed, was the crash, though he couldn\'t imagine where all the green
light came from. He couldn\'t remember his parents at all. His aunt and
uncle never spoke about them, and of course he was forbidden to ask
questions. There were no photographs of them in the house.

When he had been younger, Harry had dreamed and dreamed of some unknown
relation coming to take him away, but it had never happened; the
Dursleys were his only family. Yet sometimes he thought (or maybe hoped)
that strangers in the street seemed to know him. Very strange strangers
they were, too. A tiny man in a violet top hat had bowed to him once
while out shopping with Aunt Petunia and Dudley. After asking Harry
furiously if he knew the man, Aunt Petunia had rushed them out of the
shop without buying anything. A wild-looking old woman dressed all in
green had waved merrily at him once on a bus. A bald man in a very long
purple coat had actually shaken his hand in the street the other day and
then walked away without a word. The weirdest thing about all these
people was the way they seemed to vanish the second Harry tried to get a
closer look.

At school, Harry had no one. Everybody knew that Dudley\'s gang hated
that odd Harry Potter in his baggy old clothes and broken glasses, and
nobody liked to disagree with Dudley\'s gang.


CHAPTER THREE

THE LETTERS FROM NO ONE

The escape of the Brazilian boa constrictor earned Harry his
longest-ever punishment. By the time he was allowed out of his cupboard
again, the summer holidays had started and Dudley had already broken his
new video camera, crashed his remote control airplane, and, first time
out on his racing bike, knocked down old Mrs. Figg as she crossed Privet
Drive on her crutches.

Harry was glad school was over, but there was no escaping Dudley\'s gang,
who visited the house every single day. Piers, Dennis, Malcolm, and
Gordon were all big and stupid, but as Dudley was the biggest and
stupidest of the lot, he was the leader. The rest of them were all quite
happy to join in Dudley\'s favorite sport: Harry Hunting.

This was why Harry spent as much time as possible out of the house,
wandering around and thinking about the end of the holidays, where he
could see a tiny ray of hope. When September came he would be going off
to secondary school and, for the first time in his life, he wouldn\'t be
with Dudley. Dudley had been accepted at Uncle Vernon\'s old private
school, Smeltings. Piers Polkiss was going there too. Harry, on the
other hand, was going to Stonewall High, the local public school. Dudley
thought this was very funny.

"They stuff people\'s heads down the toilet the first day at Stonewall,"
he told Harry. "Want to come upstairs and practice?"

"No, thanks," said Harry. "The poor toilet\'s never had anything as
horrible as your head down it -- it might be sick." Then he ran, before
Dudley could work out what he\'d said.

One day in July, Aunt Petunia took Dudley to London to buy his Smeltings
uniform, leaving Harry at Mrs. Figg\'s. Mrs. Figg wasn \'t as bad as
usual. It turned out she\'d broken her leg tripping over one of her cats,
and she didn\'t seem quite as fond of them as before. She let Harry watch
television and gave him a bit of chocolate cake that tasted as though
she\'d had it for several years.

That evening, Dudley paraded around the living room for the family in
his brand-new uniform. Smeltings\' boys wore maroon tailcoats, orange
knickerbockers, and flat straw hats called boaters. They also carried
knobbly sticks, used for hitting each other while the teachers weren\'t
looking. This was supposed to be good training for later life.

As he looked at Dudley in his new knickerbockers, Uncle Vernon said
gruffly that it was the proudest moment of his life. Aunt Petunia burst
into tears and said she couldn\'t believe it was her Ickle Dudleykins, he
looked so handsome and grown-up. Harry didn\'t trust himself to speak. He
thought two of his ribs might already have cracked from trying not to
laugh.

There was a horrible smell in the kitchen the next morning when Harry
went in for breakfast. It seemed to be coming from a large metal tub in
the sink. He went to have a look. The tub was full of what looked like
dirty rags swimming in gray water.

"What\'s this?" he asked Aunt Petunia. Her lips tightened as they always
did if he dared to ask a question.

"Your new school uniform," she said.

Harry looked in the bowl again.

"Oh," he said, "I didn\'t realize it had to be so wet."

"DotA be stupid," snapped Aunt Petunia. "I\'m dyeing some of Dudley\'s old
things gray for you. It\'ll look just like everyone else\'s when I\'ve
finished."

Harry seriously doubted this, but thought it best not to argue. He sat
down at the table and tried not to think about how he was going to look
on his first day at Stonewall High -- like he was wearing bits of old
elephant skin, probably.

Dudley and Uncle Vernon came in, both with wrinkled noses because of the
smell from Harry\'s new uniform. Uncle Vernon opened his newspaper as
usual and Dudley banged his Smelting stick, which he carried everywhere,
on the table.

They heard the click of the mail slot and flop of letters on the
doormat.

"Get the mail, Dudley," said Uncle Vernon from behind his paper.

"Make Harry get it."

"Get the mail, Harry."

"Make Dudley get it."

"Poke him with your Smelting stick, Dudley."

Harry dodged the Smelting stick and went to get the mail. Three things
lay on the doormat: a postcard from Uncle Vernon\'s sister Marge, who was
vacationing on the Isle of Wight, a brown envelope that looked like a
bill, and -- a letter for Harry.

Harry picked it up and stared at it, his heart twanging like a giant
elastic band. No one, ever, in his whole life, had written to him. Who
would? He had no friends, no other relatives -- he didn\'t belong to the
library, so he\'d never even got rude notes asking for books back. Yet
here it was, a letter, addressed so plainly there could be no mistake:

Mr. H. Potter

The Cupboard under the Stairs

4 Privet Drive

Little Whinging

Surrey

The envelope was thick and heavy, made of yellowish parchment, and the
address was written in emerald-green ink. There was no stamp.

Turning the envelope over, his hand trembling, Harry saw a purple wax
seal bearing a coat of arms; a lion, an eagle, a badger, and a snake
surrounding a large letter H.

"Hurry up, boy!" shouted Uncle Vernon from the kitchen. "What are you
doing, checking for letter bombs?" He chuckled at his own joke.

Harry went back to the kitchen, still staring at his letter. He handed
Uncle Vernon the bill and the postcard, sat down, and slowly began to
open the yellow envelope.

Uncle Vernon ripped open the bill, snorted in disgust, and flipped over
the postcard.

"Marge\'s ill," he informed Aunt Petunia. "Ate a funny whelk. --."

"Dad!" said Dudley suddenly. "Dad, Harry\'s got something!"

Harry was on the point of unfolding his letter, which was written on the
same heavy parchment as the envelope, when it was jerked sharply out of
his hand by Uncle Vernon.

"That\'s mine!" said Harry, trying to snatch it back.

"Who\'d be writing to you?" sneered Uncle Vernon, shaking the letter open
with one hand and glancing at it. His face went from red to green faster
than a set of traffic lights. And it didn\'t stop there. Within seconds
it was the grayish white of old porridge.

"P-P-Petunia!" he gasped.

Dudley tried to grab the letter to read it, but Uncle Vernon held it
high out of his reach. Aunt Petunia took it curiously and read the first
line. For a moment it looked as though she might faint. She clutched her
throat and made a choking noise.

"Vernon! Oh my goodness -- Vernon!"

They stared at each other, seeming to have forgotten that Harry and
Dudley were still in the room. Dudley wasn\'t used to being ignored. He
gave his father a sharp tap on the head with his Smelting stick.

"I want to read that letter," he said loudly. want to read it," said
Harry furiously, "as it\'s mine."

"Get out, both of you," croaked Uncle Vernon, stuffing the letter back
inside its envelope.

Harry didn\'t move.

I WANT MY LETTER!" he shouted.

"Let me see it!" demanded Dudley.

"OUT!" roared Uncle Vernon, and he took both Harry and Dudley by the
scruffs of their necks and threw them into the hall, slamming the
kitchen door behind them. Harry and Dudley promptly had a furious but
silent fight over who would listen at the keyhole; Dudley won, so Harry,
his glasses dangling from one ear, lay flat on his stomach to listen at
the crack between door and floor.

"Vernon," Aunt Petunia was saying in a quivering voice, "look at the
address -- how could they possibly know where he sleeps? You don\'t think
they\'re watching the house?"

"Watching -- spying -- might be following us," muttered Uncle Vernon
wildly.

"But what should we do, Vernon? Should we write back? Tell them we don\'t
want --"

Harry could see Uncle Vernon\'s shiny black shoes pacing up and down the
kitchen.

"No," he said finally. "No, we\'ll ignore it. If they don\'t get an
answer... Yes, that\'s best... we won\'t do anything....

"But --"

"I\'m not having one in the house, Petunia! Didn\'t we swear when we took
him in we\'d stamp out that dangerous nonsense?"

That evening when he got back from work, Uncle Vernon did something he\'d
never done before; he visited Harry in his cupboard.

"Where\'s my letter?" said Harry, the moment Uncle Vernon had squeezed
through the door. "Who\'s writing to me?"

"No one. it was addressed to you by mistake," said Uncle Vernon shortly.
"I have burned it."

"It was not a mistake," said Harry angrily, "it had my cupboard on it."

"SILENCE!" yelled Uncle Vernon, and a couple of spiders fell from the
ceiling. He took a few deep breaths and then forced his face into a
smile, which looked quite painful.

"Er -- yes, Harry -- about this cupboard. Your aunt and I have been
thinking... you\'re really getting a bit big for it... we think it might
be nice if you moved into Dudley\'s second bedroom.

"Why?" said Harry.

"Don\'t ask questions!" snapped his uncle. "Take this stuff upstairs,
now."

The Dursleys\' house had four bedrooms: one for Uncle Vernon and Aunt
Petunia, one for visitors (usually Uncle Vernon\'s sister, Marge), one
where Dudley slept, and one where Dudley kept all the toys and things
that wouldn\'t fit into his first bedroom. It only took Harry one trip
upstairs to move everything he owned from the cupboard to this room. He
sat down on the bed and stared around him. Nearly everything in here was
broken. The month-old video camera was lying on top of a small, working
tank Dudley had once driven over the next door neighbor\'s dog; in the
corner was Dudley\'s first-ever television set, which he\'d put his foot
through when his favorite program had been canceled; there was a large
birdcage, which had once held a parrot that Dudley had swapped at school
for a real air rifle, which was up on a shelf with the end all bent
because Dudley had sat on it. Other shelves were full of books. They
were the only things in the room that looked as though they\'d never been
touched.

From downstairs came the sound of Dudley bawling at his mother, I don\'t
want him in there... I need that room... make him get out...."

Harry sighed and stretched out on the bed. Yesterday he\'d have given
anything to be up here. Today he\'d rather be back in his cupboard with
that letter than up here without it.

Next morning at breakfast, everyone was rather quiet. Dudley was in
shock. He\'d screamed, whacked his father with his Smelting stick, been
sick on purpose, kicked his mother, and thrown his tortoise through the
greenhouse roof, and he still didn\'t have his room back. Harry was
thinking about this time yesterday and bitterly wishing he\'d opened the
letter in the hall. Uncle Vernon and Aunt Petunia kept looking at each
other darkly.

When the mail arrived, Uncle Vernon, who seemed to be trying to be nice
to Harry, made Dudley go and get it. They heard him banging things with
his Smelting stick all the way down the hall. Then he shouted, "There\'s
another one! \'Mr. H. Potter, The Smallest Bedroom, 4 Privet Drive --\'"

With a strangled cry, Uncle Vernon leapt from his seat and ran down the
hall, Harry right behind him. Uncle Vernon had to wrestle Dudley to the
ground to get the letter from him, which was made difficult by the fact
that Harry had grabbed Uncle Vernon around the neck from behind. After a
minute of confused fighting, in which everyone got hit a lot by the
Smelting stick, Uncle Vernon straightened up, gasping for breath, with
Harry\'s letter clutched in his hand.

"Go to your cupboard -- I mean, your bedroom," he wheezed at Harry.
"Dudley -- go -- just go."

Harry walked round and round his new room. Someone knew he had moved out
of his cupboard and they seemed to know he hadn\'t received his first
letter. Surely that meant they\'d try again? And this time he\'d make sure
they didn\'t fail. He had a plan.

The repaired alarm clock rang at six o\'clock the next morning. Harry
turned it off quickly and dressed silently. He mustn\'t wake the
Dursleys. He stole downstairs without turning on any of the lights.

He was going to wait for the postman on the corner of Privet Drive and
get the letters for number four first. His heart hammered as he crept
across the dark hall toward the front door --

Harry leapt into the air; he\'d trodden on something big and squashy on
the doormat -- something alive!

Lights clicked on upstairs and to his horror Harry realized that the
big, squashy something had been his uncle\'s face. Uncle Vernon had been
lying at the foot of the front door in a sleeping bag, clearly making
sure that Harry didn\'t do exactly what he\'d been trying to do. He
shouted at Harry for about half an hour and then told him to go and make
a cup of tea. Harry shuffled miserably off into the kitchen and by the
time he got back, the mail had arrived, right into Uncle Vernon\'s lap.
Harry could see three letters addressed in green ink.

I want --" he began, but Uncle Vernon was tearing the letters into
pieces before his eyes. Uncle Vernon didnt go to work that day. He
stayed at home and nailed up the mail slot.

"See," he explained to Aunt Petunia through a mouthful of nails, "if
they can\'t deliver them they\'ll just give up."

"I\'m not sure that\'ll work, Vernon."

"Oh, these people\'s minds work in strange ways, Petunia, they\'re not
like you and me," said Uncle Vernon, trying to knock in a nail with the
piece of fruitcake Aunt Petunia had just brought him.

On Friday, no less than twelve letters arrived for Harry. As they
couldn\'t go through the mail slot they had been pushed under the door,
slotted through the sides, and a few even forced through the small
window in the downstairs bathroom.

Uncle Vernon stayed at home again. After burning all the letters, he got
out a hammer and nails and boarded up the cracks around the front and
back doors so no one could go out. He hummed "Tiptoe Through the Tulips"
as he worked, and jumped at small noises.

On Saturday, things began to get out of hand. Twenty-four letters to
Harry found their way into the house, rolled up and hidden inside each
of the two dozen eggs that their very confused milkman had handed Aunt
Petunia through the living room window. While Uncle Vernon made furious
telephone calls to the post office and the dairy trying to find someone
to complain to, Aunt Petunia shredded the letters in her food processor.

"Who on earth wants to talk to you this badly?" Dudley asked Harry in
amazement.

On Sunday morning, Uncle Vernon sat down at the breakfast table looking
tired and rather ill, but happy.

"No post on Sundays," he reminded them cheerfully as he spread marmalade
on his newspapers, "no damn letters today --"

Something came whizzing down the kitchen chimney as he spoke and caught
him sharply on the back of the head. Next moment, thirty or forty
letters came pelting out of the fireplace like bullets. The Dursleys
ducked, but Harry leapt into the air trying to catch one.

"Out! OUT!"

Uncle Vernon seized Harry around the waist and threw him into the hall.
When Aunt Petunia and Dudley had run out with their arms over their
faces, Uncle Vernon slammed the door shut. They could hear the letters
still streaming into the room, bouncing off the walls and floor.

"That does it," said Uncle Vernon, trying to speak calmly but pulling
great tufts out of his mustache at the same time. I want you all back
here in five minutes ready to leave. We\'re going away. Just pack some
clothes. No arguments!"

He looked so dangerous with half his mustache missing that no one dared
argue. Ten minutes later they had wrenched their way through the
boarded-up doors and were in the car, speeding toward the highway.
Dudley was sniffling in the back seat; his father had hit him round the
head for holding them up while he tried to pack his television, VCR, and
computer in his sports bag.

They drove. And they drove. Even Aunt Petunia didn\'t dare ask where they
were going. Every now and then Uncle Vernon would take a sharp turn and
drive in the opposite direction for a while. "Shake\'em off... shake \'em
off," he would mutter whenever he did this.

They didn\'t stop to eat or drink all day. By nightfall Dudley was
howling. He\'d never had such a bad day in his life. He was hungry, he\'d
missed five television programs he\'d wanted to see, and he\'d never gone
so long without blowing up an alien on his computer.

Uncle Vernon stopped at last outside a gloomy-looking hotel on the
outskirts of a big city. Dudley and Harry shared a room with twin beds
and damp, musty sheets. Dudley snored but Harry stayed awake, sitting on
the windowsill, staring down at the lights of passing cars and
wondering....

They ate stale cornflakes and cold tinned tomatoes on toast for
breakfast the next day. They had just finished when the owner of the
hotel came over to their table.

"\'Scuse me, but is one of you Mr. H. Potter? Only I got about an \'undred
of these at the front desk."

She held up a letter so they could read the green ink address:

Mr. H. Potter

Room 17

Railview Hotel

Cokeworth

Harry made a grab for the letter but Uncle Vernon knocked his hand out
of the way. The woman stared.

"I\'ll take them," said Uncle Vernon, standing up quickly and following
her from the dining room.

Wouldn\'t it be better just to go home, dear?" Aunt Petunia suggested
timidly, hours later, but Uncle Vernon didn\'t seem to hear her. Exactly
what he was looking for, none of them knew. He drove them into the
middle of a forest, got out, looked around, shook his head, got back in
the car, and off they went again. The same thing happened in the middle
of a plowed field, halfway across a suspension bridge, and at the top of
a multilevel parking garage.

"Daddy\'s gone mad, hasn\'t he?" Dudley asked Aunt Petunia dully late that
afternoon. Uncle Vernon had parked at the coast, locked them all inside
the car, and disappeared.


<<Second book>>

HARRY POTTER AND THE CHAMBER OF SECRETS
by J. K. Rowling

(this is BOOK 2 in the Harry Potter series)

Original Scanned/OCR: Friday, April 07, 2000
v1.0
(edit where needed, change version number by 0.1)


C H A P T E RR\t\tO N E

THE WORST BIRTHDAY

Not for the first time, an argument had broken out over breakfast at
number four, Privet Drive. Mr. Vernon Dursley had been woken in
the early hours of the morning by a loud, hooting noise from his
nephew Harry\'s room.

"Third time this week!" he roared across the table. "If you can\'t
control that owl, it\'ll have to go!"

Harry tried, yet again, to explain.

"She\'s bored," he said. "She\'s used to flying around outside. If I could
just let her out at night -"

"Do I look stupid?" snarled Uncle Vernon, a bit of fried egg dangling
from his bushy mustache. "I know what\'ll happen if that owl\'s let
out."

He exchanged dark looks with his wife, Petunia.

Harry tried to argue back but his words were drowned by a long,
loud belch from the Dursleys\' son, Dudley.

1



"I want more bacon."

"There\'s more in the frying pan, sweetums," said Aunt Petunia,
turning misty eyes on her massive son. "We must build you up while
we\'ve got the chance .... I don\'t like the sound of that school food
......

"Nonsense, Petunia, I never went hungry when I was at Smeltings,"
said Uncle Vernon heartily. "Dudley gets enough, don\'t you, son?"

Dudley, who was so large his bottom drooped over either side of the
kitchen chair, grinned and turned to Harry.

"Pass the frying pan."

"You\'ve forgotten the magic word," said Harry irritably.

The effect of this simple sentence on the rest of the family was
incredible: Dudley gasped and fell off his chair with a crash that
shook the whole kitchen; Mrs. Dursley gave a small scream and
clapped her hands to her mouth; Mr. Dursley jumped to his feet,
veins throbbing in his temples.

"I meant `please\'!" said Harry quickly. "I didn\'t mean -"

"WHAT HAVE I TOLD YOU," thundered his uncle, spraying spit
over the table, "ABOUT SAYING THE `M\' WORD IN OUR
HOUSE?"

"But I -"

"HOW DARE YOU THREATEN DUDLEY!" roared Uncle
Vernon, pounding the table with his fist.

"I just -"

"I WARNED YOU! I WILL NOT TOLERATE MENTION OF
YOUR ABNORMALITY UNDER THIS ROOF!"

Harry stared from his purple-faced uncle to his pale aunt, who was
trying to heave Dudley to his feet.

"All right," said Harry, "all right. . . "

Uncle Vernon sat back down, breathing like a winded rhinoceros and
watching Harry closely out of the corners of his small, sharp eyes.

Ever since Harry had come home for the summer holidays, Uncle
Vernon had been treating him like a bomb that might go off at any
moment, because Harry Potter wasn\'t a normal boy. As a matter of
fact, he was as not normal as it is possible to be.

Harry Potter was a wizard - a wizard fresh from his first year at
Hogwarts School of Witchcraft and Wizardry. And if the Dursleys
were unhappy to have him back for the holidays, it was nothing to how
Harry felt.

He missed Hogwarts so much it was like having a constant
stomachache. He missed the castle, with its secret passageways and
ghosts, his classes (though perhaps not Snape, the Potions master), the
mail arriving by owl, eating banquets in the Great Hall, sleeping in his
four-poster bed in the tower dormitory, visiting the gamekeeper,
Hagrid, in his cabin next to the Forbidden Forest in the grounds, and,
especially, Quidditch, the most popular sport in the wizarding world
(six tall goal posts, four flying balls, and fourteen players on
broomsticks).

All Harry\'s spellbooks, his wand, robes, cauldron, and top-of-the-line
Nimbus Two Thousand broomstick had been locked in a cupboard
under the stairs by Uncle Vernon the instant Harry had come home.
What did the Dursleys care if Harry lost his place on the House
Quidditch team because he hadn\'t practiced all summer? What was it
to the Dursleys if Harry went back to school without any of his
homework done? The Dursleys were what wizards called Muggles
(not a drop of magical blood in their veins),

and as far as they were concerned, having a wizard in the family was
a matter of deepest shame. Uncle Vernon had even padlocked
Harry\'s owl, Hedwig, inside her cage, to stop her from carrying
messages to anyone in the wizarding world.

Harry looked nothing like the rest of the family. Uncle Vernon was
large and neckless, with an enormous black mustache; Aunt Petunia
was horse-faced and bony; Dudley was blond, pink, and porky. Harry,
on the other hand, was small and skinny, with brilliant green eyes and
jet-black hair that was always untidy. He wore round glasses, and on
his forehead was a thin, lightning-shaped scar.

It was this scar that made Harry so particularly unusual, even for a
wizard. This scar was the only hint of Harry\'s very mysterious past, of
the reason he had been left on the Dursleys\' doorstep eleven years
before.

At the age of one year old, Harry had somehow survived a curse from
the greatest Dark sorcerer of all time, Lord Voldemort, whose name
most witches and wizards still feared to speak. Harry\'s parents had
died in Voldemort\'s attack, but Harry had escaped with his lightning
scar, and somehow - nobody understood why Voldemort\'s powers had
been destroyed the instant he had failed to kill Harry.

So Harry had been brought up by his dead mother\'s sister and her
husband. He had spent ten years with the Dursleys, never
understanding why he kept making odd things happen without meaning
to, believing the Dursleys\' story that he had got his scar in the car
crash that had killed his parents.

And then, exactly a year ago, Hogwarts had written to Harry,

and the whole story had come out. Harry had taken up his place at
wizard school, where he and his scar were famous ... but now the
school year was over, and he was back with the Dursleys for the
summer, back to being treated like a dog that had rolled in something
smelly.

The Dursleys hadn\'t even remembered that today happened to be
Harry\'s twelfth birthday. Of course, his hopes hadn\'t been high; they\'d
never given him a real present, let alone a cake - but to ignore it
completely ...

At that moment, Uncle Vernon cleared his throat importantly and said,
"Now, as we all know, today is a very important day."

Harry looked up, hardly daring to believe it.

"This could well be the day I make the biggest deal of my career, "
said Uncle Vernon.

\tHarry went back to his toast. Of course, he thought bitterly, Un
cle Vernon was talking about the stupid dinner party. He\'d been talk
ing of nothing else for two weeks. Some rich builder and his wife
were coming to dinner and Uncle Vernon was hoping to get a huge
order from him (Uncle Vernon\'s company made drills).

"I think we should run through the schedule one more time," said
Uncle Vernon. "We should all be in position at eight o\'clock. Petunia,
you will be -?"

"In the lounge," said Aunt Petunia promptly, "waiting to welcome them
graciously to our home."

"Good, good. And Dudley?"

"I\'ll be waiting to open the door." Dudley put on a foul, simpering
smile. "May I take your coats, Mr. and Mrs. Mason?"

"They\'ll love him!" cried Aunt Petunia rapturously.

"Excellent, Dudley," said Uncle Vernon. Then he rounded on Harry.
"And you?"

"I\'ll be in my bedroom, making no noise and pretending I\'m not
there," said Harry tonelessly.

"Exactly," said Uncle Vernon nastily. "I will lead them into the
lounge, introduce you, Petunia, and pour them -drinks. At eight-
fifteen -"

"I\'ll announce dinner," said Aunt Petunia.

"And, Dudley, you\'ll say -"

"May I take you through to the dining room, Mrs. Mason?" said
Dudley, offering his fat arm to an invisible woman.

"My perfect little gentleman!" sniffed Aunt Petunia.

"And you?" said Uncle Vernon viciously to Harry.

"I\'ll be in my room, making no noise and pretending I\'m not there,"
said Harry dully.

"Precisely. Now, we should aim to get in a few good compliments at
dinner. Petunia, any ideas?"

"Vernon tells me you\'re a wonderful golfer, Mr. Mason.... Do tell me
where you bought your dress, Mrs. Mason ......

"Perfect. . . Dudley?"

"How about -\'We had to write an essay about our hero at school,
Mr. Mason, and I wrote about you."\'

This was too much for both Aunt Petunia and Harry. Aunt Petunia
burst into tears and hugged her son, while Harry ducked under the
table so they wouldn\'t see him laughing.

"And you, boy?"

Harry fought to keep his face straight as he emerged.

"I\'ll be in my room, making no noise and pretending I\'m not there,"
he said.

"Too right, you will," said Uncle Vernon forcefully. "The Ma
sons don\'t know anything about you and it\'s going to stay that way.
When dinner\'s over, you take Mrs. Mason back to the lounge for
coffee, Petunia, and I\'ll bring the subject around to drills. With any
luck, I\'ll have the deal signed and sealed before the news at ten.
be shopping for a vacation home in Majorca this time to
morrow.
Harry couldn\'t feel too excited about this. He didn\'t think the
Dursleys would like him any better in Majorca than they did on
Privet Drive.
"Right - I\'m off into town to pick up the dinner jackets for
Dudley and me. And you," he snarled at Harry. "You stay out of
your aunt\'s way while she\'s cleaning."
Harry left through the back door. It was a brilliant, sunny day.
He crossed the lawn, slumped down on the garden bench, and sang
under his breath:
"Happy birthday to me ... happy birthday to me. . .
No cards, no presents, and he would be spending the evening
pretending not to exist. He gazed miserably into the hedge. He had
never felt so lonely. More than anything else at Hogwarts, more
even than playing Quidditch, Harry missed his best friends, Ron
Weasley and Hermione Granger. They, however, didn\'t seem to be
missing him at all. Neither of them had written to him all summer,
even though Ron had said he was going to ask Harry to come and
stay.
Countless times, Harry had been on the point of unlocking
Hedwig\'s cage by magic and sending her to Ron and Hermione
with a letter, but it wasn\'t worth the risk. Underage wizards weren\'t
allowed to use magic outside of school. Harry hadn\'t told the

Dursleys this; he knew it was only their terror that he might turn them
all into dung beetles that stopped them from locking him in the
cupboard under the stairs with his wand and broomstick. For the first
couple of weeks back, Harry had enjoyed muttering nonsense words
under his breath and watching Dudley tearing out of the room as fast
as his fat legs would carry him. But the long silence from Ron and
Hermione had made Harry feel so cut off from the magical world that
even taunting Dudley had lost its appeal - and now Ron and Hermione
had forgotten his birthday.

What wouldn\'t he give now for a message from Hogwarts? From any
witch or wizard? He\'d almost be glad of a sight of his archenemy,
Draco Malfoy, just to be sure it hadn\'t all been a dream ....

Not that his whole year at Hogwarts had been fun. At the very end of
last term, Harry had come face-to-face with none other than Lord
Voldemort himself. Voldemort might be a ruin of his former self, but
he was still terrifying, still cunning, still determined to regain power.
Harry had slipped through Voldemort\'s clutches for a second time, but
it had been a narrow escape, and even now, weeks later, Harry kept
waking in the night, drenched in cold sweat, wondering where
Voldemort was now, remembering his livid face, his wide, mad eyes

Harry suddenly sat bolt upright on the garden bench. He had been
staring absent-mindedly into the hedge - and the hedge was staring back.
Two enormous green eyes had appeared among the leaves.

Harry jumped to his feet just as a jeering voice floated across the
lawn.

"I know what day it is," sang Dudley, waddling toward him.

The huge eyes blinked and vanished.

"What?" said Harry, not taking his eyes off the spot where they had
been.

"I know what day it is," Dudley repeated, coming right up to him.

"Well done," said Harry. "So you\'ve finally learned the days of the
week."

"Today\'s your birthday," sneered Dudley. "How come you haven\'t got
any cards? Haven\'t you even got friends at that freak place?"

"Better not let your mum hear you talking about my school," said
Harry coolly.

Dudley hitched up his trousers, which were slipping down his fat
bottom.

"Why\'re you staring at the hedge?" he said suspiciously.

\t" I , m trying to decide what would be the best spell to set it on
fire," said Harry.

Dudley stumbled backward at once, a look of panic on his fat face.

"You c-can\'t - Dad told you you\'re not to do m-magic - he said he\'ll
chuck you out of the house - and you haven\'t got anywhere else to go -
you haven\'t got any friends to take you -"

"Jiggery pokery!" said Harry in a fierce voice. "Hocus pocus squiggly
wiggly -"

"MUUUUUUM!" howled Dudley, tripping over his feet as he dashed
back toward the house. "MUUUUM! He\'s doing you know what!"

Harry paid dearly for his moment of fun. As neither Dudley nor

the hedge was in any way hurt, Aunt Petunia knew he hadn\'t really
done magic, but he still had to duck as she aimed a heavy blow at his
head with the soapy frying pan. Then she gave him work to do, with
the promise he wouldn\'t eat again until he\'d finished.

While Dudley lolled around watching and eating ice cream, Harry
cleaned the windows, washed the car, mowed the lawn, trimmed the
flowerbeds, pruned and watered the roses, and repainted the garden
bench. The sun blazed overhead, burning the back of his neck. Harry
knew he shouldn\'t have risen to Dudley\'s bait, but Dudley had said
the very thing Harry had been thinking himself... maybe he didn\'t have
any friends at Hogwarts ....

Wish they could see famous Harry Potter now, he thought savagely as he
spread manure on the flower beds, his back aching, sweat running
down his face.

It was half past seven ,in the evening when at last, exhausted, he
heard Aunt Petunia calling him.

"Get in here! And walk on the newspaper!"

Harry moved gladly into the shade of the gleaming kitchen. On top of
the fridge stood tonight\'s pudding: a huge mound of whipped cream
and sugared violets. A loin of roast pork was sizzling in the oven.

"Eat quickly! The Masons will be here soon!" snapped Aunt Petunia,
pointing to two slices of bread and a lump of cheese on the kitchen
table. She was already wearing a salmon-pink cocktail dress.

Harry washed his hands and bolted down his pitiful supper. The
moment he had finished, Aunt Petunia whisked away his plate.
"Upstairs! Hurry!"

As he passed the door to the living room, Harry caught a
glimpse of Uncle Vernon and Dudley in bow ties and dinner jack
ets. He had only just reached the upstairs landing when the door
bell rang and Uncle Vernon\'s furious face appeared at the foot of
the stairs.
"Remember, boy - one sound -"
Harry crossed to his bedroom on tiptoe slipped inside, closed
the door, and turned to collapse on his bed.
The trouble was, there was already someone sitting on it.

C H-H A P T E RR\t\tT W o

I

DOBBY\'S WARNING

arry managed not to shout out, but it was a close thing. The little
creature on the bed had large, bat-like ears and bulging green eyes the
size of tennis balls. Harry knew instantly that this was what had been
watching him out of the garden hedge that morning.

As they stared at each other, Harry heard Dudley\'s voice from the hall.

"May I take your coats, Mr. and Mrs. Mason?"

The creature slipped off the bed and bowed so low that the end of its
long, thin nose touched the carpet. Harry noticed that it was wearing
what looked like an old pillowcase, with rips for arm- and leg-holes.

"Er - hello," said Harry nervously.

"Harry Potter!" said the creature in a high-pitched voice Harry was
sure would carry down the stairs. "So long has Dobby wanted to meet
you, sir ... Such an honor it is . . . ."



"Th-thank you," said Harry, edging along the wall and sinking into his
desk chair, next to Hedwig, who was asleep in her large cage. He
wanted to ask, "What are you?" but thought it would sound too rude,
so instead he said, "Who are you?"

"Dobby, sir. Just Dobby. Dobby the house-elf," said the creature.

"Oh - really?" said Harry. "Er - I don\'t want to be rude or anything,
but - this isn\'t a great time for me to have a house-elf in my
bedroom."

Aunt Petunias high, false laugh sounded from the living room. The elf
hung his head.

"Not that I\'m not pleased to meet you," said Harry quickly, "but, er,
is there any particular reason you\'re here?"

"Oh, yes, sir," said Dobby earnestly. "Dobby has come to tell you,
sir ... it is difficult, sir ... Dobby wonders where to begin . . . ."

"Sit down," said Harry politely, pointing at the bed.

To his horror, the elf burst into tears - very noisy tears.

"S-sit down!" he wailed. "Never ... never ever. . . "

Harry thought he heard the voices downstairs falter.

"I\'m sorry," he whispered, "I didn\'t mean to offend you or anything -"

"Offend Dobby!" choked the elf. "Dobby has never been asked to sit
down by a wizard - like an equal-"

Harry, trying to say "Shh!" and look comforting at the same time,
ushered Dobby back onto the bed where he sat hiccoughing, looking
like a large and very ugly doll. At last he managed to control himself,
and sat with his great eyes fixed on Harry in an expression of watery
adoration.

"You can\'t have met many decent wizards," said Harry, trying to
cheer him up.

Dobby shook his head. Then, without warning, he leapt up and
started banging his head furiously on the window, shouting, "Bad
Dobby! Bad Dobby!"

"Don\'t - what are you doing?" Harry hissed, springing up and pulling
Dobby back onto the bed - Hedwig had woken up with a
particularly loud screech and was beating her wings wildly against the
bars of her cage.

"Dobby had to punish himself, sir," said the elf, who had gone slightly
cross-eyed. "Dobby almost spoke ill of his family, sir . . . ."

"Your family?"

"The wizard family Dobby serves, sir... DOBBY\'S is a houseelf -
bound to serve one house and one family forever . .....

"Do they know you\'re here?" asked Harry curiously.

Dobby shuddered.

"Oh, no, sir, no ... Dobby will have to punish himself most grievously
for coming to see you, sir. Dobby will have to shut his ears in the
oven door for this. If they ever knew, sir _"

"But won\'t they notice if you shut your ears in the oven door?"

"Dobby doubts it, sir. Dobby is always having to punish himself for
something, sir. They lets Dobby get on with it, sir. Sometimes they
reminds me to do extra punishments ......

"But why don\'t you leave? Escape?"

"A house-elf must be set free, sir. And the family will never set
Dobby free ... Dobby will serve the family until he dies, sir . . . ."

Harry stared.

"And I thought I had it bad staying here for another four weeks,"

he said. "This makes the Dursleys sound almost human. Can\'t anyone
help you? Can\'t I?"

Almost at once, Harry wished he hadn\'t spoken. Dobby dissolved again
into wails of gratitude.

"Please," Harry whispered frantically, "please be quiet. If the Dursleys
hear anything, if they know you\'re here -"

"Harry Potter asks if he can help Dobby ... Dobby has heard of your
greatness, sir, but of your goodness, Dobby never knew . .....

Harry, who was feeling distinctly hot in the face, said, "Whatever
you\'ve heard about my greatness is a load of rubbish. I\'m not even top
of my year at Hogwarts; that\'s Hermione, she -"

But he stopped quickly, because thinking about Hermione was painful.

"I-Tarry Potter is humble and modest," said Dobby reverently, his orb-
like eyes aglow. "Harry Potter speaks not of his triumph over He-Who-
Must-Not-Be-Named -"

"Voldemort?" said Harry.

Dobby clapped his hands over his bat ears and moaned, "Ah, speak not
the name, sir! Speak not the name!"

"Sorry" said Harry quickly. "I know lots of people don\'t like it. My
friend Ron -"

He stopped again. Thinking about Ron was painful, too.

Dobby leaned toward Harry, his eyes wide as headlights.

\'Dobby heard tell," he said hoarsely, "that Harry Potter met the Dark
Lord for a second time just weeks ago ... that Harry Potter escaped
Yet again. "

Harry nodded and Dobby\'s eyes suddenly shone with tears.

,Ah, sir," he gasped, dabbing his face with a corner of the grubby

pillowcase he was wearing. "Harry Potter is valiant and bold! He has
braved so many dangers already! But Dobby has come to protect
Harry Potter, to warn him, even if he does have to shut his ears in
the oven door later... Harry Potter must notgo back to Hogwarts."

There was a silence broken only by the chink of knives and forks
from downstairs and the distant rumble of Uncle Vernon\'s voice.

"W-what?" Harry stammered. "But I\'ve got to go back - term starts
on September first. It\'s all that\'s keeping me going. You don\'t know
what it\'s like here. I don\'t belong here. I belong in your world - at
Hogwarts."

"No, no, no," squeaked Dobby, shaking his head so hard his ears
flapped. "Harry Potter must stay where he is safe. He is too great,
too good, to lose. If Harry Potter goes back to Hogwarts, he will be
in mortal danger."

"Why?" said Harry in surprise.

"There is a plot, Harry Potter. A plot to make most terrible things
happen at Hogwarts School of Witchcraft and Wizardry this year,"
whispered Dobby, suddenly trembling all over. "Dobby has known it
for months, sir. Harry Potter must not put himself in peril. He is too
important, sir!"

"What terrible things?" said Harry at once. "Who\'s plotting them?"

Dobby made a funny choking noise and then banged his head
frantically against the wall.

"All right!" cried Harry, grabbing the elf\'s arm to stop him. "You can\'t
tell me. I understand. But why are you warning me?" A sudden,
unpleasant thought struck him. "Hang on - this hasn\'t got anything to
do with Vol- - sorry - with You-Know-Who, has it?

You could just shake or nod," he added hastily as Dobby\'s head
tilted worryingly close to the wall again.

Slowly, Dobby shook his head.

"Not -not He- Who-Must-Not-Be-Named, sir =\'

But Dobby\'s eyes were wide and he seemed to be trying to give
Harry a hint. Harry, however, was completely lost.

"He hasn\'t got a brother, has he?"

Dobby shook his head, his eyes wider than ever.

"Well then, I can\'t think who else would have a chance of making
horrible things happen at Hogwarts," said Harry. "I mean, there\'s
Dumbledore, for one thing - you know who Dumbledore is, don\'t
you?"

Dobby bowed his head.

"Albus Dumbledore is the greatest headmaster Hogwarts has ever
had. Dobby knows it, sir. Dobby has heard Dumbledore\'s powers
rival those of He-Who-Must-Not-Be-Named at the height of his
strength. But, sir" - Dobby\'s voice dropped to an urgent whisper -
"there are powers Dumbledore doesn\'t ... powers no decent wizard.
. ."

And before Harry could stop him, Dobby bounded off the bed,
seized Harry\'s desk lamp, and started beating himself around the
head with earsplitting yelps.

A sudden silence fell downstairs. Two seconds later Harry, heart
thudding madly, heard Uncle Vernon coming into the hall, calling,
"Dudley must have left his television on again, the little tyke!"

"Quick! In the closet!" hissed Harry, stuffing Dobby in, shutting the
door, and flinging himself onto the bed just as the door handle turned.

"What - the - devil - are - you - doing?" said Uncle Vernon through
gritted teeth, his face horribly close to Harry\'s. "You\'ve just ruined the
punch line of my Japanese golfer joke .... One more sound and you\'ll
wish you\'d never been born, boy!"

He stomped flat-footed from the room.

Shaking, Harry let Dobby out of the closet.

"See what it\'s like here?" he said. "See why I\'ve got to go back to
Hogwarts? It\'s the only place I\'ve got -well, I think I\'ve got friends. "

"Friends who don\'t even write to Harry Potter?" said Dobby slyly.

"I expect they\'ve just been - wait a minute," said Harry, frowning.
"How do you know my friends haven\'t been writing to me?"

Dobby shuffled his feet.

"Harry Potter mustn\'t be angry with Dobby. Dobby did it for the best -
"

"Have you been stopping my letters?"

"Dobby has them here, sir," said the elf. Stepping nimbly out of Harry\'s
reach, he pulled a thick wad of envelopes from the inside of the
pillowcase he was wearing. Harry could make out Hermione\'s neat
writing, Ron\'s untidy scrawl, and even a scribble that looked as though
it was from the Hogwarts gamekeeper, Hagrid.

Dobby blinked anxiously up at Harry.

"Harry Potter mustn\'t be angry... Dobby hoped ... if Harry Potter
thought his friends had forgotten him ... Harry Potter might not want to
go back to school, sir . .....

Harry wasn\'t listening. He made a grab for the letters, but Dobby
jumped out of reach.

"Harry Potter will have them, sir, if he gives Dobby his word

that he will not return to Hogwarts. Ah, sir, this is a danger you must
not face! Say you won\'t go back, sir!"

"No," said Harry angrily. "Give me my friends\' letters!"

"Then Harry Potter leaves Dobby no choice," said the elf sadly.

Before Harry could move, Dobby had darted to the bedroom door,
pulled it open, and sprinted down the stairs.

Mouth dry, stomach lurching, Harry sprang after him, trying not to
make a sound. He jumped the last six steps, landing catlike on the
hall carpet, looking around for Dobby. From the dining room he
heard Uncle Vernon saying, ". . . tell Petunia that very funny story
about those American plumbers, Mr. Mason. She\'s been dying to
hear. . . "

Harry ran up the hall into the kitchen and felt his stomach disappear.

Aunt Petunia\'s masterpiece of a pudding, the mountain of cream and
sugared violets, was floating up near the ceiling. On top of a
cupboard in the corner crouched Dobby.

"No," croaked Harry. "Please ... they\'ll kill me ......

"Harry Potter must say he\'s not going back to school -"

"Dobby ... please ...

"Say it, sir -"

"I can\'t -"

Dobby gave him a tragic look.

"Then Dobby must do it, sir, for Harry Potter\'s own good."

The pudding fell to the floor with a heart-stopping crash. Cream
splattered the windows and walls as the dish shattered. With a crack
like a whip, Dobby vanished.

There were screams from the dining room and Uncle Vernon

burst into the kitchen to find Harry, rigid with shock, covered from head
to foot in Aunt Petunias pudding.

At first, it looked as though Uncle Vernon would manage to gloss the
whole thing over. ("Just our nephew - very disturbed

\tmeeting strangers upsets him, so we kept him upstairs \t) He

shooed the shocked Masons back into the dining room, promised
Harry he would flay him to within an inch of his life when the Ma
sons had left, and handed him a mop. Aunt Petunia dug some ice
cream out of the freezer and Harry, still shaking, started scrubbing
the kitchen clean.

Uncle Vernon might still have been able to make his deal - if it hadn\'t
been for the owl.

Aunt Petunia was just passing around a box of after-dinner mints when
a huge barn owl swooped through the dining room window, dropped a
letter on Mrs. Mason\'s head, and swooped out again. Mrs. Mason
screamed like a banshee and ran from the house shouting about
lunatics. Mr. Mason stayed just long enough to tell the Dursleys that his
wife was mortally afraid of birds of all shapes and sizes, and to ask
whether this was their idea of a joke.

Harry stood in the kitchen, clutching the mop for support, as Uncle
Vernon advanced on him, a demonic glint in his tiny eyes.

"Read it!" he hissed evilly, brandishing the letter the owl had delivered.
"Go on - read it!"

Harry took it. It did not contain birthday greetings.

Dear Mr. Potter,

We have received intelligence that a Hover Charm was used at your
place of residence this evening at twelve minutes past nine.

As you know, underage wizards are not permitted to perform spells
outside school, and further spellwork on your part may lead to
expulsion from said school (Decree for the Reasonable Restriction of
Underage Sorcery, 1875, Paragraph C).

We would also ask you to remember that any magical activity that
risks notice by members of the non-magical community (Muggles) is
a serious offense under section 13 of the International Confederation
of Warlocks\' Statute of Secrecy.

Enjoy your holidays! Yours sincerely,

Mafalda Hopkirk

IMPROPER USE OF MAGIC OFFICE

Ministry of Magic

Harry looked up from the letter and gulped.

"You didn\'t tell us you weren\'t allowed to use magic outside school,"
said Uncle Vernon, a mad gleam dancing in his eyes. "For got to
mention it .... Slipped your mind, I daresay .....

He was bearing down on Harry like a great bulldog, all his teeth
bared. "Well, I\'ve got news for you, boy . ... I\'m locking you up ....
You\'re never going back to that school ... never ... and if you try and
magic yourself out - they\'ll expel you!"

And laughing like a maniac, he dragged Harry back upstairs.

Uncle Vernon was as bad as his word. The following morning,



he paid a man to fit bars on Harry\'s window. He himself fitted a cat-
flap in the bedroom door, so that small amounts of food could be
pushed inside three times a day. They let Harry out to use the
bathroom morning and evening. Otherwise, he was locked in his room
around the clock.

Three days later, the Dursleys were showing no sign of relenting, and
Harry couldn\'t see any way out of his situation. He lay on his bed
watching the sun sinking behind the bars on the window and wondered
miserably what was going to happen to him.

What was the good of magicking himself out of his room if Hogwarts
would expel him for doing it? Yet life at Privet Drive had reached an
all-time low. Now that the Dursleys knew they weren\'t going to wake
up as fruit bats, he had lost his only weapon. Dobby might have saved
Harry from horrible happenings at Hogwarts, but the way things were
going, he\'d probably starve to death anyway.

The cat-flap rattled and Aunt Petunias hand appeared, pushing a bowl
of canned soup into the room. Harry, whose insides were aching with
hunger, jumped off his bed and seized it. The soup was stone-cold, but
he drank half of it in one gulp. Then he crossed the room to Hedwig\'s
cage and tipped the soggy vegetables at the bottom of the bowl into
her empty food tray. She ruffled her feathers and gave him a look of
deep disgust.

"It\'s no good turning your beak up at it - that\'s all we\'ve got," said
Harry grimly.

He put the empty bowl back on the floor next to the cat-flap and lay
back down on the bed, somehow even hungrier than he had been
before the soup.

Supposing he was still alive in another four weeks, what would happen
if he didn\'t turn up at Hogwarts? Would someone be sent to see why
he hadn\'t come back? Would they be able to make the Dursleys let
him go?

The room was growing dark. Exhausted, stomach rumbling, mind
spinning over the same unanswerable questions, Harry fell into an
uneasy sleep.

He dreamed that he was on show in a zoo, with a card reading
UNDERAGE WIZARD attached to his cage. People goggled through the bars
at him as he lay, starving and weak, on a bed of straw. He saw
Dobby\'s face in the crowd and shouted out, asking for help, but Dobby
called, "Harry Potter is safe there, sir!" and vanished. Then the
Dursleys appeared and Dudley rattled the bars of the cage, laughing at
him.

"Stop it," Harry muttered as the rattling pounded in his sore head.
"Leave me alone ... cut it out ... I\'m trying to sleep . . . ."

He opened his eyes. Moonlight was shining through the bars on the
window. And someone was goggling through the bars at him: a freckle-
faced, red-haired, long-nosed someone.

Ron Weasley was outside Harry\'s window.

H-H A P T E RR T 11-H RR E E

THE BURROW

Ron.l" breathed Harry, creeping to the window and pushing it up so
they could talk through the bars. "Ron, how did you - What the -?"

Harry\'s mouth fell open as the full impact of what he was seeing hit
him. Ron was leaning out of the back window of an old turquoise car,
which was parked in midair Grinning at Harry from the front seats
were Fred and George, Ron\'s elder twin brothers.

"All right, Harry?" asked George.

"What\'s been going on?" said Ron. "Why haven\'t you been answering
my letters? I\'ve asked you to stay about twelve times, and then Dad
came home and said you\'d got an official warning for using magic in
front of Muggles -"

"It wasn\'t me - and how did he know?"

"He works for the Ministry," said Ron. "You know we\'re not supposed
to do spells outside school -"



"You should talk," said Harry, staring at the floating car.

"Oh, this doesn\'t count," said Ron. "We\'re only borrowing this. It\'s
Dad\'s, we didn\'t enchant it. But doing magic in front of those Muggles
you live with -"

"I told you, I didn\'t - but it\'ll take too long to explain now look, can you
tell them at Hogwarts that the Dursleys have locked me up and won\'t
let me come back, and obviously I can\'t magic myself out, because the
Ministry\'Il think that\'s the second spell I\'ve done in three days, so -"

"Stop gibbering," said Ron. "We\'ve come to take you home with us."

"But you can\'t magic me out either -"

"We don\'t need to," said Ron, jerking his head toward the front seat
and grinning. "You forget who I\'ve got with me."

"Tie that around the bars," said Fred, throwing the end of a rope to
Harry.

"If the Dursleys wake up, I\'m dead," said Harry as he tied the rope
tightly around a bar and Fred revved up the car.

"Don\'t worry," said Fred, "and stand back."

Harry moved back into the shadows next to Hedwig, who seemed to
have realized how important this was and kept still and silent. The car
revved louder and louder and suddenly, with a crunching noise, the
bars were pulled clean out of the window as Fred drove straight up in
the air. Harry ran back to the window to see the bars dangling a few
feet above the ground. Panting, Ron hoisted them up into the car.
Harry listened anxiously, but there was no sound from the Dursleys\'
bedroom.

When the bars were safely in the back seat with Ron, Fred reversed
as close as possible to Harry\'s window.

"Get in," Ron said.

"But all my Hogwarts stuff - my wand - my broomstick -"

"Where is it?"

"Locked in the cupboard under the stairs, and I can\'t get out of this
room -"

"No problem," said George from the front passenger seat. "Out of
the way, Harry."

Fred and George climbed catlike through the window into Harry\'s
room. You had to hand it to them, thought Harry, as George took an
ordinary hairpin from his pocket and started to pick the lock.

"A lot of wizards think it\'s a waste of time, knowing this sort of
Muggle trick," said Fred, "but we feel they\'re skills worth learning,
even if they are a bit slow."

There was a small click and the door swung open.

"So - we\'ll get your trunk - you grab anything you need from your
room and hand it out to Ron," whispered George.

"Watch out for the bottom stair - it creaks," Harry whispered back
as the twins disappeared onto the dark landing.

Harry dashed around his room, collecting his things and passing them
out of the window to Ron. Then he went to help Fred and George
heave his trunk up the stairs. Harry heard Uncle Vernon cough.

At last, panting, they reached the landing, then carried the trunk
through Harry\'s room to the open window. Fred climbed back into
the car to pull with Ron, and Harry and George pushed from the
bedroom side. Inch by inch, the trunk slid through the window.

Uncle Vernon coughed again.

"A bit more," panted Fred, who was pulling from inside the car.
"One good push -"

Harry and George threw their shoulders against the trunk and it slid
out of the window into the back seat of the car.

"Okay, let\'s go," George whispered.

But as Harry climbed onto the windowsill there came a sudden loud
screech from behind him, followed immediately by the thunder of
Uncle Vernon\'s voice.

"THAT RUDDY OWL!"

"I\'ve forgotten Hedwig!"

Harry tore back across the room as the landing light clicked on - he
snatched up Hedwig\'s cage, dashed to the window, and passed it
out to Ron. He was scrambling back onto the chest of drawers when
Uncle Vernon hammered on the unlocked door and it crashed open.

For a split second, Uncle Vernon stood framed in the doorway; then
he let out a bellow like an angry bull and dived at Harry, grabbing
him by the ankle.

Ron, Fred, and George seized Harry\'s arms and pulled as hard as
they could.

"Petunia!" roared Uncle Vernon. "He\'s getting away! HE\'S
GETTING AWAY!"

But the Weasleys gave a gigantic tug and Harry\'s leg slid out of
Uncle Vernon\'s grasp - Harry was in the car - he\'d slammed the
door shut

"Put your foot down, Fred!" yelled Ron, and the car shot suddenly
toward the moon.

Harry couldn\'t believe it - he was free. He rolled down the

window, the night air whipping his hair, and looked back at the
shrinking rooftops of Privet Drive. Uncle Vernon, Aunt Petunia, and
Dudley were all hanging, dumbstruck, out of Harry\'s window.

"See you next summer!" Harry yelled.

The Weasleys roared with laughter and Harry settled back in his seat,
grinning from ear to ear.

"Let Hedwig out," he told Ron. "She can fly behind us. She hasn\'t had
a chance to stretch her wings for ages."

George handed the hairpin to Ron and, a moment later, Hedwig soared
joyfully out of the window to glide alongside them like a ghost.

"So - what\'s the story, Harry?" said Ron impatiently. "What\'s been
happening?"

Harry told them all about Dobby, the warning he\'d given Harry and
the fiasco of the violet pudding. There was a long, shocked silence
when he had finished.

"Very fishy," said Fred finally.

"Definitely dodgy" agreed George. "So he wouldn\'t even tell you who\'s
supposed to be plotting all this stuff?"

"I don\'t think he could," said Harry. "I told you, every time he got close
to letting something slip, he started banging his head against the wall."

He saw Fred and George look at each other.

"What, you think he was lying to me?" said Harry.

"Well," said Fred, "put it this way - house-elves have got powerful
magic of their own, but they can\'t usually use it without their master\'s
permission. I reckon old Dobby was sent to stop you com

ing back to Hogwarts. Someone\'s idea of a joke. Can you think of
anyone at school with a grudge against you?"

"Yes," said Harry and Ron together, instantly.

"Draco Malfoy," Harry explained. "He hates me."

"Draco Malfoy?" said George, turning around. "Not Lucius Malfoy\'s
son?"

"Must be, it\'s not a very common name, is it?" said Harry.

Y.

"I\'ve heard Dad talking about him," said George. "He was a big
supporter of You-Know-Who."

"And when You-Know-Who disappeared," said Fred, craning
around to look at Harry, "Lucius Malfoy came back saying he\'d never
meant any of it. Load of dung - Dad reckons he was right in You-
Know-Who\'s inner circle."

Harry had heard these rumors about Malfoy\'s family before, and they
didn\'t surprise him at all. Malfoy made Dudley Dursley look

\tlike a kind, thoughtful, and sensitive boy.
\t"I don\'t know whether the Malfoys own a house-elf \tsaid
\tHarry.

"Well, whoever owns him will be an old wizarding family, and they\'ll
be rich," said Fred.

"Yeah, Mum\'s always wishing we had a house-elf to do the ironing,"
said George. "But all we\'ve got is a lousy old ghoul in the attic and
gnomes all over the garden. House-elves come with big old manors
and castles and places like that; you wouldn\'t catch one in our house .
. . ."

Harry was silent. Judging by the fact that Draco Malfoy usually had
the best of everything, his family was rolling in wizard gold; he

could just see Malfoy strutting around a large manor house. Sending
the family servant to stop Harry from going back to Hogwarts also
sounded exactly like the sort of thing Malfoy would do. Had Harry
been stupid to take Dobby seriously?

"I\'m glad we came to get you, anyway," said Ron. "I was getting
really worried when you didn\'t answer any of my letters. I thought it
was Errol\'s fault at first

-"

"Who\'s Errol?"

"Our owl. He\'s ancient. It wouldn\'t be the first time he\'d collapsed
on a delivery. So then I tried to borrow Hermes -"

"Who?"

"The owl Mum and Dad bought Percy when he was made prefect,"
said Fred from the front.

"But Percy wouldn\'t lend him to me," said Ron. "Said he needed
him."

"Percy\'s been acting very oddly this summer," said George,
frowning. "And he has been sending a lot of letters and spending a
load of time shut up in his room .... I mean, there\'s only so many
times you can polish a prefect badge .... You\'re driving too far west,
Fred," he added, pointing at a compass on the dashboard. Fred
twiddled the steering wheel.

"So, does your dad know you\'ve got the car?" said Harry, guessing
the answer.

"Er, no," said Ron, "he had to work tonight. Hopefully we\'ll be able
to get it back in the garage without Mum noticing we flew it."

"What does your dad do at the Ministry of Magic, anyway?"

"He works in the most boring department," said Ron. "The Misuse
of Muggle Artifacts Office."

"The what?"

"It\'s all to do with bewitching things that are Muggle-made, you
know, in case they end up back in a Muggle shop or house. Like,
last year, some old witch died and her tea set was sold to an antiques
shop. This Muggle woman bought it, took it home, and tried to serve
her friends tea in it. It was a nightmare - Dad was working overtime
for weeks."

"What happened?"

"The teapot went berserk and squirted boiling tea all over the place
and one man ended up in the hospital with the sugar tongs clamped
to his nose. Dad was going frantic - it\'s only him and an old warlock
called Perkins in the office -and they had to do Memory Charms and
all sorts of stuff to cover it up -"

"But your dad - this car -"

Fred laughed. "Yeah, Dad\'s crazy about everything to do with
Muggles; our shed\'s full of Muggle stuff. He takes it apart, puts spells
on it, and puts it back together again. If he raided our house he\'d
have to put himself under arrest. It drives Mum mad."

"That\'s the main road," said George, peering down through the
windshield. "We\'ll be there in ten minutes .... Just as well, it\'s getting
light . . . ."

A faint pinkish glow was visible along the horizon to the east.

Fred brought the car lower, and Harry saw a dark patchwork of
fields and clumps of trees.

"We\'re a little way outside the village," said George. "Ottery St.
Catchpole."

Lower and lower went the flying car. The edge of a brilliant red sun
was now gleaming through the trees.

"Touchdown!" said Fred as, with a slight bump, they hit the ground.
They had landed next to a tumbledown garage in a small yard, and
Harry looked out for the first time at Ron\'s house.

It looked as though it had once been a large stone pigpen, but extra
rooms had been added here and there until it was several stories high
and so crooked it looked as though it were held up by magic (which,
Harry reminded himself, it probably was). Four or five chimneys were
perched on top of the red roof. A lopsided sign stuck in the ground
near the entrance read, THE BuRRow. Around the front door lay a jumble
of rubber boots and a very rusty cauldron. Several fat brown chickens
were pecking their way around the yard.

"It\'s not much," said Ron.

"It\'s wonderful," said Harry happily, thinking of Privet Drive.

They got out of the car.

"Now, we\'ll go upstairs really quietly," said Fred, "and wait for Mum to
call us for breakfast Then, Ron, you come bounding downstairs going,
`Mum, look who turned up in the night!\' and she\'ll be all pleased to see
Harry and no one need ever know we flew the car."

"Right," said Ron. "Come on, Harry, I sleep at the - at the top

Ron had gone a nasty greenish color, his eyes fixed on the house. The
other three wheeled around.

Mrs. Weasley was marching across the yard, scattering chickens, and
for a short, plump, kind-faced woman, it was remarkable how much
she looked like a saber-toothed tiger.

"Ah, "said Fred.

"Oh, dear," said George.

Mrs. Weasley came to a halt in front of them, her hands on her hips,
staring from one guilty face to the next. She was wearing a flowered
apron with a wand sticking out of the pocket.

"So, "she said.

"Morning, Mum," said George, in what he clearly thought was a jaunty,
winning voice.

"Have you any idea how worried I\'ve been?" said Mrs. Weasley in a
deadly whisper.

"Sorry, Mum, but see, we had to -"

All three of Mrs. Weasley\'s sons were taller than she was, but they
cowered as her rage broke over them.

"Beds empty! No note! Cargone - could have crashed - out of my

mind with worry - did you care? - never, as long as I\'ve lived -
you wait until your father gets home, we never had trouble like this
from Bill or Charlie or Percy -"

"Perfect Percy," muttered Fred.

"YOU COULD DO WITH TAKING A LEAF OUT OF PERCY\'S
BOOK!" yelled Mrs. Weasley, prodding a finger in Fred\'s chest. "You
could have died, you could have been seen, you could have lost your
father his job -"

It seemed to go on for hours. Mrs. Weasley had shouted herself
hoarse before she turned on Harry, who backed away.

"I\'m very pleased to see you, Harry, dear," she said. "Come in and
have some breakfast."

She turned and walked back into the house and Harry, after a nervous
glance at Ron, who nodded encouragingly, followed her.

The kitchen was small and rather cramped. There was a

scrubbed wooden table and chairs in the middle, and Harry sat down
on the edge of his seat, looking around. He had never been in a wizard
house before.

The clock on the wall opposite him had only one hand and no numbers
at all. Written around the edge were things like Time to make tea, Time
to feed the chickens, and You\'re late. Books were stacked three deep on
the mantelpiece, books with titles like Charm Your Own Cheese,
Enchantment in Baking, and One Minute Feasts - It\'s Magic! And unless
Harry\'s ears were deceiving him, the old radio next to the sink had just
announced that coming up was "Witching Hour, with the popular
singing sorceress, Celestina Warbeck."

Mrs. Weasley was clattering around, cooking breakfast a little
haphazardly, throwing dirty looks at her sons as she threw sausages
into the frying pan. Every now and then she muttered things like "don\'t
know what you were thinking of," and "never would have believed it."

"I don\'t blame you, dear," she assured Harry, tipping eight or nine
sausages onto his plate. "Arthur and I have been worried about you,
too. Just last night we were saying we\'d come and get you ourselves if
you hadn\'t written back to Ron by Friday. But really," (she was now
adding three fried eggs to his plate) "flying an illegal car halfway
across the country - anyone could have seen you -"

She flicked her wand casually at the dishes in the sink, which began to
clean themselves, clinking gently in the background.

"It was cloudy, Mum!" said Fred.

"You keep your mouth closed while you\'re eating!" Mrs. Weasley
snapped.

"They were starving him, Mum!" said George.

"And you!" said Mrs. Weasley, but it was with a slightly softened
expression that she started cutting Harry bread and buttering it for
him.

At that moment there was a diversion in the form of a small,
redheaded figure in a long nightdress, who appeared in the kitchen,
gave a small squeal, and ran out again.

"Ginny," said Ron in an undertone to Harry. "My sister. She\'s been
talking about you all summer."

"Yeah, she\'ll be wanting your autograph, Harry," Fred said with a grin,
but he caught his mother\'s eye and bent his face over his plate without
another word. Nothing more was said until all four plates were clean,
which took a surprisingly short time.

"Blimey, I\'m tired," yawned Fred, setting down his knife and fork at last.
"I think I\'ll go to bed and -"

"You will not," snapped Mrs. Weasley. "It\'s your own fault you\'ve
been up all night. You\'re going to de-gnome the garden for me; they\'re
getting completely out of hand again -"

"Oh, Mum -"

"And you two," she said, glaring at Ron and Fred. "You can go up to
bed, dear," she added to Harry. "You didn\'t ask them to fly that
wretched car -"

But Harry, who felt wide awake, said quickly, "I\'ll help Ron. I\'ve
never seen a de-gnoming -"

"That\'s very sweet of you, dear, but it\'s dull work," said Mrs. Weasley.
"Now, let\'s see what Lockhart\'s got to say on the subject -"

And she pulled a heavy book from the stack on the mantelpiece.
George groaned.

"Mum, we know how to de-gnome a garden -"

Harry looked at the cover of Mrs. Weasley\'s book. Written across it
in fancy gold letters were the words Gilderoy Lockhart\'s Guide to
Household Pests. There was a big photograph on the front of a very good-
IOI)king wizard with wavy blond hair and bright blue eyes. As always
in the wizarding world, the photograph was moving; the wizard, who
Harry supposed was Gilderoy Lockhart, kept winking cheekily up at
them all. Mrs. Weasley beamed down at him.

"Oh, he is marvelous," she said. "He knows his household pests, all
right, it\'s a wonderful book . . . ."

"Mum fancies him," said Fred, in a very audible whisper.

"Don\'t be so ridiculous, Fred," said Mrs. Weasley, her cheeks rather
pink. "All right, if you think you know better than Lockhart, you can go
and get on with it, and woe betide you if there\'s a single gnome in that
garden when I come out to inspect it."

Yawning and grumbling, the Weasleys slouched outside with Harry
behind them. The garden was large, and in Harry\'s eyes, exactlY
what a garden should be. The Dursleys wouldn\'t have liked it - there
were plenty of weeds, and the grass needed cutting but there were
gnarled trees all around the walls, plants Harry had never seen spilling
from every flower bed, and a big green pond full of frogs.

"Muggles have garden gnomes, too, you know," Harry told Ron

they crossed the lawn.

"Yeah, I\'ve seen those things they think are gnomes," said Ron, bent
double with his head in a peony bush, "like fat little Santa Clauses with
fishing rods . . . ."

There was a violent scuffling noise, the peony bush shuddered, and
Ron straightened up. "This is a gnome," he said grimly.

"Gerroff me! Gerroff me!" squealed the gnome.

It was certainly nothing like Santa Claus. It was small and leathery
looking, with a large, knobby, bald head exactly like a potato. Ron held
it at arm\'s length as it kicked out at him with its horny little feet; he
grasped it around the ankles and turned it upside down.

"This is what you have to do," he said. He raised the gnome above his
head ("Gerroff me!") and started to swing it in great circles like a
lasso. Seeing the shocked look on Harry\'s face, Ron added, "It doesn\'t
hurt them - you\'ve just got to make them really dizzy so they can\'t find
their way back to the gnomeholes."

He let go of the gnome\'s ankles: It flew twenty feet into the air and
landed with a thud in the field over the hedge.

"Pitiful," said Fred. "I bet I can get mine beyond that stump."

Harry learned quickly not to feel too sorry for the gnomes. He decided
just to drop the first one he caught over the hedge, but the gnome,
sensing weakness, sank its razor-sharp teeth into Harry\'s finger and he
had a hard job shaking it off - until

"Wow, Harry - that must\'ve been fifty feet ......

The air was soon thick with flying gnomes.

"See, they\'re not too bright," said George, seizing five or six gnomes at
once. "The moment they know the de-gnoming\'s going on they storm
up to have a look. You\'d think they\'d have learned by now just to stay
put."

Soon, the crowd of gnomes in the field started walking away in a
straggling line, their little shoulders hunched.

"They\'ll be back," said Ron as they watched the gnomes disappear into
the hedge on the other side of the field. "They love it here .... Dad\'s
too soft with them; he thinks they\'re funny . . . ."

Just then, the front door slammed.

"He\'s back!" said George. "Dad\'s home!"

They hurried through the garden and back into the house.

Mr. Weasley was slumped in a kitchen chair with his glasses off and
his eyes closed. He was a thin man, going bald, but the little hair he
had was as red as any of his children\'s. He was wearing long green
robes, which were dusty and travel-worn.

"What a night," he mumbled, groping for the teapot as they all sat
down around him. "Nine raids. Nine! And old Mundungus Fletcher
tried to put a hex on me when I had my back turned ......

Mr. Weasley took a long gulp of tea and sighed.

"Why would anyone bother making door keys which it is very shrink?" said George.

"Just Muggle-baiting," sighed Mr. Weasley. "Sell them a key that
keeps shrinking to nothing so they can never find it when they need it
.... Of course, it\'s very hard to convict anyone because no Muggle
would admit their key keeps shrinking - they\'ll insist they just keep
losing it. Bless them, they\'ll go to any lengths to ignore magic, even if
it\'s staring them in the face .... But the things our lot have taken to
enchanting, you wouldn\'t believe -"

"LIKE CARS, FOR INSTANCE?"

Mrs. Weasley had appeared, holding a long poker like a sword of it.

"C-cars, Molly, dear?"

"Yes, Arthur, cars," said Mrs. Weasley, her eyes flashing. "Imagine a
wizard buying a rusty old car and telling his wife all he wanted to do
with it was take it apart to see how it worked, while really he was
enchanting it to make it fly."

Mr. Weasley blinked.

"Well, dear, I think you\'ll find that he would be quite within the law to
do that, even if - er - he maybe would have done better to, um, tell his
wife the truth .... There\'s a loophole in the law, you\'ll find .... As long
as he wasn\'t intending to fly the car, the fact that the car could fly
wouldn\'t -"

"Arthur Weasley, you made sure there was a loophole when you
wrote that law!" shouted Mrs. Weasley. "Just so you could carry on
tinkering with all that Muggle rubbish in your shed! And for your
information, Harry arrived this morning in the car you weren\'t
intending to fly!"

"Harry?" said Mr. Weasley blankly. "Harry who?"

He looked around, saw Harry, and jumped.

"Good lord, is ight."

shouted Mrs. Weasley. "What have you got to say about that, eh?"

"Did you really?" said Mr. Weasley eagerly. "Did it go all right? I - I
mean," he faltered as sparks flew from Mrs. Weasley\'s eyes, "that -
that was very wrong, boys - very wrong indeed ......

"Let\'s leave them to it," Ron muttered to Harry as Mrs. Weasley
swelled like a bullfrog. "Come on, I\'ll show you my bedroom."

They slipped out of the kitchen and down a narrow passageway to an
uneven staircase, which wound its way, zigzagging up

through the house. On the third landing, a door stood ajar. Harry just
caught sight of a pair of bright brown eyes staring at him before it
closed with a snap.

"Ginny," said Ron. "You don\'t know how weird it is for her to be this
shy. She never shuts up normally -"

They climbed two more flights until they reached a door with peeling
paint and a small plaque on it, saying RONALD\'S ROOM.

"Your Quidditch team?" said Harry.

"The Chudley Cannons," said Ron, pointing at the orange bedspread,
which was emblazoned with two giant black C\'s and a speeding
cannonball. "Ninth in the league."

Ron\'s school spellbooks were stacked untidily in a corner, next to a
pile of comics that all seemed to feature The Adventures of Martin
Miggs, the Mad Muggle. Ron\'s magic wand was lying on top of a fish
tank full of frog spawn on the windowsill, next to his fat gray rat,
Scabbers, who was snoozing in a patch of sun.

Harry stepped over a pack of Self-Shuffling playing cards on the floor
and looked out of the tiny window. In the field far below he could see
a gang of gnomes sneaking one by one back through the Weasleys\'
hedge. Then he turned to look at Ron, who was watching him almost
nervously, as though waiting for his opinion.

"It\'s a bit small," said Ron quickly. "Not like that room you had
with the Muggles. And I\'m right underneath the ghoul in the attic;
he\'s always banging on the pipes and groaning ......
But Harry, grinning widely, said, "This is the best house I\'ve ever
been in."
Ron\'s ears went pink. .

C H4 A P T E R\t\tV O U R

AT F L 0 V RR 11 $ H
AND BLOTTS

ife at the Burrow was as different as possible from life on Privet
Drive. The Dursleys liked everything neat and ordered; the Weasleys\'
house burst with the strange and unexpected. Harry got a shock the
first time he looked in the mirror over the kitchen mantelpiece and it
shouted, "Tuck your shirt in, scruffy!" The ghoul in the attic howled
and dropped pipes whenever he felt things were getting too quiet, and
small explosions from Fred and George\'s bedroom were considered
perfectly normal. What Harry found most unusual about life at Ron\'s,
however, wasn\'t the talking mirror or the clanking ghoul: It was the
fact that everybody there seemed to like him.

Mrs. Weasley fussed over the state of his socks and tried to force him
to eat fourth helpings at every meal. Mr. Weasley liked Harry to sit
next to him at the dinner table so that he could bombard him with
questions about life with Muggles, asking him to explain how things
like plugs and the postal service worked.

Ron had gone a nasty greenish color, his eyes fixed on the house. The
other three wheeled around.

Mrs. Weasley was marching across the yard, scattering chickens, and
for a short, plump, kind-faced woman, it was remarkable how much
she looked like a saber-toothed tiger.

"Ah, "said Fred.

"Oh, dear," said George.

Mrs. Weasley came to a halt in front of them, her hands on her hips,
staring from one guilty face to the next. She was wearing a flowered
apron with a wand sticking out of the pocket.

"So, "she said.

"Morning, Mum," said George, in what he clearly thought was a jaunty,
winning voice.

"Have you any idea how worried I\'ve been?" said Mrs. Weasley in a
deadly whisper.

"Sorry, Mum, but see, we had to -"

All three of Mrs. Weasley\'s sons were taller than she was, but they
cowered as her rage broke over them.

"Beds empty! No note! Cargone - could have crashed - out of my

mind with worry - did you care? - never, as long as I\'ve lived -
you wait until your father gets home, we never had trouble like this
from Bill or Charlie or Percy -"

"Perfect Percy," muttered Fred.

"YOU COULD DO WITH TAKING A LEAF OUT OF PERCY\'S
BOOK!" yelled Mrs. Weasley, prodding a finger in Fred\'s chest. "You
could have died, you could have been seen, you could have lost your
father his job -"

It seemed to go on for hours. Mrs. Weasley had shouted herself
hoarse before she turned on Harry, who backed away.

She flicked her wand casually at the dishes in the sink, which began to
clean themselves, clinking gently in the background.

"It was cloudy, Mum!" said Fred.

"You keep your mouth closed while you\'re eating!" Mrs. Weasley
snapped.

"They were starving him, Mum!" said George.

"And you!" said Mrs. Weasley, but it was with a slightly softened
expression that she started cutting Harry bread and buttering it for
him.

At that moment there was a diversion in the form of a small,
redheaded figure in a long nightdress, who appeared in the kitchen,
gave a small squeal, and ran out again.

"Ginny," said Ron in an undertone to Harry. "My sister. She\'s been
talking about you all summer."

"Yeah, she\'ll be wanting your autograph, Harry," Fred said with a grin,
but he caught his mother\'s eye and bent his face over his plate without
another word. Nothing more was said until all four plates were clean,
which took a surprisingly short time.

"Blimey, I\'m tired," yawned Fred, setting down his knife and fork at last.
"I think I\'ll go to bed and -"

"You will not," snapped Mrs. Weasley. "It\'s your own fault you\'ve
been up all night. You\'re going to de-gnome the garden for me; they\'re
getting completely out of hand again -"

"Oh, Mum -"

"And you two," she said, glaring at Ron and Fred. "You can go up to
bed, dear," she added to Harry. "You didn\'t ask them to fly that
wretched car -"

But Harry, who felt wide awake, said quickly, "I\'ll help Ron. I\'ve
never seen a de-gnoming -"

"That\'s very sweet of you, dear, but it\'s dull work," said Mrs. Weasley.
"Now, let\'s see what Lockhart\'s got to say on the subject -"

And she pulled a heavy book from the stack on the mantelpiece.
George groaned.

"Mum, we know how to de-gnome a garden -"

Harry looked at the cover of Mrs. Weasley\'s book. Written across it
in fancy gold letters were the words Gilderoy Lockhart\'s Guide to
Household Pests. There was a big photograph on the front of a very good-
IOI)king wizard with wavy blond hair and bright blue eyes. As always
in the wizarding world, the photograph was moving; the wizard, who
Harry supposed was Gilderoy Lockhart, kept winking cheekily up at
them all. Mrs. Weasley beamed down at him.

"Oh, he is marvelous," she said. "He knows his household pests, all
right, it\'s a wonderful book . . . ."

"Mum fancies him," said Fred, in a very audible whisper.

"Don\'t be so ridiculous, Fred," said Mrs. Weasley, her cheeks rather
pink. "All right, if you think you know better than Lockhart, you can go
and get on with it, and woe betide you if there\'s a single gnome in that
garden when I come out to inspect it."

Yawning and grumbling, the Weasleys slouched outside with Harry
behind them. The garden was large, and in Harry\'s eyes, exactlY
what a garden should be. The Dursleys wouldn\'t have liked it - there
were plenty of weeds, and the grass needed cutting but there were
gnarled trees all around the walls, plants Harry had never seen spilling
from every flower bed, and a big green pond full of frogs.

"Muggles have garden gnomes, too, you know," Harry told Ron

they crossed the lawn.

"Yeah, I\'ve seen those things they think are gnomes," said Ron, bent
double with his head in a peony bush, "like fat little Santa Clauses with
fishing rods . . . ."


Summarization:
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
        # "The capital of France is",
        # "The capital of the United Kindom is",
        # "Today is a sunny day and I like",
        # "Sky is blue because",
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
