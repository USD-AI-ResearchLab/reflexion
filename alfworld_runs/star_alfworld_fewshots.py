"""
STAR-format fewshots for ALFWorld.

Format mirrors hotpotqa_runs/fewshots.py WEBTHINK_SIMPLE2:
  - LLM outputs: THOUGHT / ACTION / EXPECTED / KEY / optional CORRECTION
  - Environment injects: Observation N:
  - CORRECTION appears on the step AFTER a mismatch (prev EXPECTED != observation)
  - Scratchpad only stores action/observation lines (STAR fields not in scratchpad)

Two examples per task type (pick-and-place and heat-and-place) following the
same 2-shot structure as the standard alfworld_3prompts.json.
"""

# ---------------------------------------------------------------------------
# pick-and-place (put)
# ---------------------------------------------------------------------------

STAR_PUT_1 = """Task: put a apple in fridge.
THOUGHT: I need to find an apple. It could be on the countertop, table, or in the fridge already. Let me check the countertop first.
ACTION: go to countertop 1
EXPECTED: I will see objects on the countertop, possibly including an apple.
KEY: go-to-countertop-search
Observation 1: On the countertop 1, you see a apple 1, a bread 1, and a knife 1.
THOUGHT: I found an apple on the countertop. Now I need to pick it up.
ACTION: take apple 1 from countertop 1
EXPECTED: I will pick up the apple successfully.
KEY: take-object-from-surface
Observation 2: You pick up the apple 1.
THOUGHT: I have the apple. Now I need to put it in the fridge.
ACTION: go to fridge 1
EXPECTED: I will arrive at the fridge.
KEY: go-to-target-receptacle
Observation 3: You arrive at the fridge 1. The fridge 1 is closed.
THOUGHT: The fridge is closed. I need to open it before placing the apple inside.
ACTION: open fridge 1
EXPECTED: The fridge will open and I can see inside.
KEY: open-receptacle-before-place
CORRECTION: Receptacles like fridge are often closed — always open them before trying to place an object inside.
Observation 4: You open the fridge 1. The fridge 1 is now open.
THOUGHT: The fridge is open. Now I can put the apple inside.
ACTION: put apple 1 in/on fridge 1
EXPECTED: The apple will be placed in the fridge and the task will be complete.
KEY: put-object-in-receptacle
Observation 5: You put the apple 1 in/on the fridge 1."""

STAR_PUT_0 = """Task: put a mug in cabinet.
THOUGHT: I need to find a mug. Mugs are often on the countertop, table, or in a cabinet. Let me check the countertop.
ACTION: go to countertop 1
EXPECTED: I will see the countertop and find a mug there.
KEY: go-to-countertop-search
Observation 1: On the countertop 1, you see a cup 1, a plate 1.
THOUGHT: No mug on countertop 1. Let me check countertop 2.
ACTION: go to countertop 2
EXPECTED: I will find a mug on countertop 2.
KEY: go-to-countertop-search
CORRECTION: When the first countertop does not have the target object, check other countertops before moving to other furniture types.
Observation 2: On the countertop 2, you see a mug 1, a spatula 2.
THOUGHT: Found a mug on countertop 2. I will pick it up.
ACTION: take mug 1 from countertop 2
EXPECTED: I will pick up the mug successfully.
KEY: take-object-from-surface
Observation 3: You pick up the mug 1.
THOUGHT: I have the mug. Now I need to put it in a cabinet. Let me go to cabinet 1.
ACTION: go to cabinet 1
EXPECTED: I will arrive at cabinet 1.
KEY: go-to-target-receptacle
Observation 4: You arrive at cabinet 1. The cabinet 1 is closed.
THOUGHT: The cabinet is closed. I need to open it first.
ACTION: open cabinet 1
EXPECTED: Cabinet 1 will open.
KEY: open-receptacle-before-place
Observation 5: You open the cabinet 1.
THOUGHT: Cabinet is open. Now I can put the mug inside.
ACTION: put mug 1 in/on cabinet 1
EXPECTED: The mug will be placed in the cabinet and the task will be complete.
KEY: put-object-in-receptacle
Observation 6: You put the mug 1 in/on the cabinet 1."""


# ---------------------------------------------------------------------------
# heat-and-place (heat)
# ---------------------------------------------------------------------------

STAR_HEAT_1 = """Task: heat a potato and put it in fridge.
THOUGHT: I need to find a potato first. Potatoes can be on countertops or in the fridge. Let me check the countertop.
ACTION: go to countertop 1
EXPECTED: I will see objects on the countertop, possibly including a potato.
KEY: go-to-countertop-search
Observation 1: On the countertop 1, you see a tomato 1 and a knife 1.
THOUGHT: No potato on countertop 1. Let me check countertop 2.
ACTION: go to countertop 2
EXPECTED: I will find a potato on countertop 2.
KEY: go-to-countertop-search
Observation 2: On the countertop 2, you see a potato 1 and a spatula 1.
THOUGHT: Found a potato. I need to pick it up and heat it in the microwave.
ACTION: take potato 1 from countertop 2
EXPECTED: I will pick up the potato.
KEY: take-object-from-surface
Observation 3: You pick up the potato 1.
THOUGHT: I have the potato. I need to heat it in the microwave.
ACTION: go to microwave 1
EXPECTED: I will arrive at the microwave.
KEY: go-to-microwave-heat
Observation 4: You arrive at the microwave 1. The microwave 1 is closed.
THOUGHT: Microwave is closed. I need to open it, place the potato, then heat it.
ACTION: open microwave 1
EXPECTED: The microwave will open.
KEY: open-appliance-before-use
Observation 5: You open the microwave 1.
THOUGHT: Microwave is open. I'll put the potato in it.
ACTION: put potato 1 in/on microwave 1
EXPECTED: The potato will be placed inside the microwave.
KEY: put-object-in-appliance
Observation 6: You put the potato 1 in/on the microwave 1.
THOUGHT: Now I need to heat the potato using the microwave.
ACTION: heat potato 1 with microwave 1
EXPECTED: The potato will be heated successfully.
KEY: heat-object-with-appliance
Observation 7: You heat the potato 1 with the microwave 1.
THOUGHT: The potato is heated. Now I need to take it out and put it in the fridge.
ACTION: take potato 1 from microwave 1
EXPECTED: I will pick up the heated potato.
KEY: take-object-from-appliance
Observation 8: You pick up the potato 1.
THOUGHT: I have the heated potato. Now I need to put it in the fridge.
ACTION: go to fridge 1
EXPECTED: I will arrive at the fridge.
KEY: go-to-target-receptacle
Observation 9: You arrive at the fridge 1. The fridge 1 is closed.
THOUGHT: Fridge is closed. I need to open it.
ACTION: open fridge 1
EXPECTED: The fridge will open.
KEY: open-receptacle-before-place
Observation 10: You open the fridge 1.
THOUGHT: Fridge is open. I can now put the potato inside.
ACTION: put potato 1 in/on fridge 1
EXPECTED: The task will be complete.
KEY: put-object-in-receptacle
Observation 11: You put the potato 1 in/on the fridge 1."""

STAR_HEAT_0 = """Task: heat a egg and put it in countertop.
THOUGHT: I need to find an egg. Eggs are often in the fridge or on the countertop. Let me check the fridge.
ACTION: go to fridge 1
EXPECTED: I will open the fridge and find an egg inside.
KEY: go-to-fridge-search
Observation 1: You arrive at the fridge 1. The fridge 1 is closed.
THOUGHT: The fridge is closed. I need to open it.
ACTION: open fridge 1
EXPECTED: The fridge will open and I will see an egg inside.
KEY: open-receptacle-before-search
Observation 2: You open the fridge 1. Inside the fridge 1, you see an egg 1 and a lettuce 1.
THOUGHT: Found an egg in the fridge. Let me take it.
ACTION: take egg 1 from fridge 1
EXPECTED: I will pick up the egg.
KEY: take-object-from-receptacle
Observation 3: You pick up the egg 1.
THOUGHT: I have the egg. I need to heat it in the microwave.
ACTION: go to microwave 1
EXPECTED: I will arrive at the microwave.
KEY: go-to-microwave-heat
Observation 4: You arrive at the microwave 1. The microwave 1 is open.
THOUGHT: Microwave is already open. I can put the egg inside directly.
ACTION: put egg 1 in/on microwave 1
EXPECTED: The egg will be placed inside the microwave.
KEY: put-object-in-appliance
Observation 5: You put the egg 1 in/on the microwave 1.
THOUGHT: Now I heat the egg.
ACTION: heat egg 1 with microwave 1
EXPECTED: The egg will be heated.
KEY: heat-object-with-appliance
Observation 6: You heat the egg 1 with the microwave 1.
THOUGHT: Egg is heated. I need to take it out and place it on the countertop.
ACTION: take egg 1 from microwave 1
EXPECTED: I will pick up the heated egg.
KEY: take-object-from-appliance
Observation 7: You pick up the egg 1.
THOUGHT: I have the heated egg. Now I put it on the countertop.
ACTION: go to countertop 1
EXPECTED: I will arrive at the countertop.
KEY: go-to-target-receptacle
Observation 8: You arrive at the countertop 1.
THOUGHT: I am at the countertop. I can now place the egg here.
ACTION: put egg 1 in/on countertop 1
EXPECTED: The task will be complete.
KEY: put-object-in-receptacle
Observation 9: You put the egg 1 in/on the countertop 1."""


# ---------------------------------------------------------------------------
# clean-and-place (clean)
# ---------------------------------------------------------------------------

STAR_CLEAN_1 = """Task: clean a cup and put it in cabinet.
THOUGHT: I need to find a cup. Cups are typically on the countertop or in a cabinet. Let me check the countertop.
ACTION: go to countertop 1
EXPECTED: I will see objects on the countertop, possibly a cup.
KEY: go-to-countertop-search
Observation 1: On the countertop 1, you see a cup 1, a fork 1.
THOUGHT: Found a cup. I need to take it and clean it in the sink.
ACTION: take cup 1 from countertop 1
EXPECTED: I will pick up the cup.
KEY: take-object-from-surface
Observation 2: You pick up the cup 1.
THOUGHT: I have the cup. I need to clean it in the sink or sinkbasin.
ACTION: go to sinkbasin 1
EXPECTED: I will arrive at the sinkbasin.
KEY: go-to-sink-clean
Observation 3: You arrive at the sinkbasin 1.
THOUGHT: I am at the sinkbasin. I can now clean the cup.
ACTION: clean cup 1 with sinkbasin 1
EXPECTED: The cup will be cleaned.
KEY: clean-object-with-sink
Observation 4: You clean the cup 1 with the sinkbasin 1.
THOUGHT: Cup is clean. Now I need to put it in a cabinet.
ACTION: go to cabinet 1
EXPECTED: I will arrive at cabinet 1.
KEY: go-to-target-receptacle
Observation 5: You arrive at cabinet 1. The cabinet 1 is closed.
THOUGHT: Cabinet is closed. I must open it first.
ACTION: open cabinet 1
EXPECTED: The cabinet will open.
KEY: open-receptacle-before-place
Observation 6: You open the cabinet 1.
THOUGHT: Cabinet is open. I can put the cup inside.
ACTION: put cup 1 in/on cabinet 1
EXPECTED: The task will be complete.
KEY: put-object-in-receptacle
Observation 7: You put the cup 1 in/on the cabinet 1."""

STAR_CLEAN_0 = """Task: clean a plate and put it in shelf.
THOUGHT: I need to find a plate. Let me search the countertop.
ACTION: go to countertop 1
EXPECTED: I will find a plate on the countertop.
KEY: go-to-countertop-search
Observation 1: On the countertop 1, you see a plate 1 and a cup 2.
THOUGHT: Found a plate. I'll pick it up and clean it.
ACTION: take plate 1 from countertop 1
EXPECTED: I will pick up the plate.
KEY: take-object-from-surface
Observation 2: You pick up the plate 1.
THOUGHT: I have the plate. I need to clean it at the sink.
ACTION: go to sink 1
EXPECTED: I will arrive at the sink.
KEY: go-to-sink-clean
Observation 3: Nothing happens.
THOUGHT: The action failed — the correct object name might be sinkbasin not sink.
ACTION: go to sinkbasin 1
EXPECTED: I will arrive at the sinkbasin.
KEY: go-to-sink-clean
CORRECTION: The correct action target is sinkbasin not sink — always use sinkbasin for cleaning tasks.
Observation 4: You arrive at the sinkbasin 1.
THOUGHT: I am at the sinkbasin. I can clean the plate now.
ACTION: clean plate 1 with sinkbasin 1
EXPECTED: The plate will be cleaned.
KEY: clean-object-with-sink
Observation 5: You clean the plate 1 with the sinkbasin 1.
THOUGHT: Plate is clean. Now I need to put it on the shelf.
ACTION: go to shelf 1
EXPECTED: I will arrive at the shelf.
KEY: go-to-target-receptacle
Observation 6: You arrive at the shelf 1.
THOUGHT: I am at the shelf. I can place the plate here.
ACTION: put plate 1 in/on shelf 1
EXPECTED: The task will be complete.
KEY: put-object-in-receptacle
Observation 7: You put the plate 1 in/on the shelf 1."""


# ---------------------------------------------------------------------------
# cool-and-place (cool)
# ---------------------------------------------------------------------------

STAR_COOL_1 = """Task: cool a tomato and put it in shelf.
THOUGHT: I need to find a tomato. Tomatoes are typically on the countertop. Let me check.
ACTION: go to countertop 1
EXPECTED: I will find a tomato on the countertop.
KEY: go-to-countertop-search
Observation 1: On the countertop 1, you see a tomato 1, a knife 1.
THOUGHT: Found a tomato. I'll pick it up and cool it in the fridge.
ACTION: take tomato 1 from countertop 1
EXPECTED: I will pick up the tomato.
KEY: take-object-from-surface
Observation 2: You pick up the tomato 1.
THOUGHT: I have the tomato. I need to cool it in the fridge.
ACTION: go to fridge 1
EXPECTED: I will arrive at the fridge.
KEY: go-to-fridge-cool
Observation 3: You arrive at the fridge 1. The fridge 1 is closed.
THOUGHT: Fridge is closed. I need to open it.
ACTION: open fridge 1
EXPECTED: The fridge will open.
KEY: open-receptacle-before-use
Observation 4: You open the fridge 1.
THOUGHT: Fridge is open. I'll cool the tomato using the fridge.
ACTION: cool tomato 1 with fridge 1
EXPECTED: The tomato will be cooled.
KEY: cool-object-with-fridge
Observation 5: You cool the tomato 1 with the fridge 1.
THOUGHT: Tomato is cooled. Now I need to put it on the shelf.
ACTION: go to shelf 1
EXPECTED: I will arrive at the shelf.
KEY: go-to-target-receptacle
Observation 6: You arrive at the shelf 1.
THOUGHT: I am at the shelf. I can place the tomato here.
ACTION: put tomato 1 in/on shelf 1
EXPECTED: The task will be complete.
KEY: put-object-in-receptacle
Observation 7: You put the tomato 1 in/on the shelf 1."""

STAR_COOL_0 = """Task: cool a lettuce and put it in countertop.
THOUGHT: I need to find a lettuce. Let me check the fridge first.
ACTION: go to fridge 1
EXPECTED: I will find a lettuce in the fridge.
KEY: go-to-fridge-search
Observation 1: You arrive at the fridge 1. The fridge 1 is closed.
THOUGHT: Fridge is closed. I need to open it first.
ACTION: open fridge 1
EXPECTED: The fridge will open and I will see a lettuce inside.
KEY: open-receptacle-before-search
Observation 2: You open the fridge 1. Inside the fridge 1, you see a tomato 2 and a apple 1.
THOUGHT: No lettuce in the fridge. Let me check the countertop.
ACTION: go to countertop 1
EXPECTED: I will find a lettuce on the countertop.
KEY: go-to-countertop-search
CORRECTION: Not all items are in the fridge — when the fridge is empty of the target, check countertops and tables before giving up.
Observation 3: On the countertop 1, you see a lettuce 1 and a spatula 1.
THOUGHT: Found a lettuce on the countertop. I need to cool it in the fridge.
ACTION: take lettuce 1 from countertop 1
EXPECTED: I will pick up the lettuce.
KEY: take-object-from-surface
Observation 4: You pick up the lettuce 1.
THOUGHT: I have the lettuce. I need to cool it using the fridge.
ACTION: go to fridge 1
EXPECTED: I will arrive at the fridge which is already open.
KEY: go-to-fridge-cool
Observation 5: You arrive at the fridge 1. The fridge 1 is open.
THOUGHT: Fridge is already open. I can cool the lettuce now.
ACTION: cool lettuce 1 with fridge 1
EXPECTED: The lettuce will be cooled.
KEY: cool-object-with-fridge
Observation 6: You cool the lettuce 1 with the fridge 1.
THOUGHT: Lettuce is cooled. Now I need to put it on the countertop.
ACTION: go to countertop 2
EXPECTED: I will arrive at the countertop.
KEY: go-to-target-receptacle
Observation 7: You arrive at the countertop 2.
THOUGHT: I am at the countertop. I can place the lettuce here.
ACTION: put lettuce 1 in/on countertop 2
EXPECTED: The task will be complete.
KEY: put-object-in-receptacle
Observation 8: You put the lettuce 1 in/on the countertop 2."""


# ---------------------------------------------------------------------------
# examine-in-light (examine)
# ---------------------------------------------------------------------------

STAR_EXAMINE_1 = """Task: examine a book with a desklamp.
THOUGHT: I need to find a book and a desklamp. Let me first look for a book.
ACTION: go to sidetable 1
EXPECTED: I will see objects on the sidetable, possibly including a book.
KEY: go-to-sidetable-search
Observation 1: On the sidetable 1, you see a book 1, a pen 1.
THOUGHT: Found a book. Now I need to find a desklamp to examine it under.
ACTION: take book 1 from sidetable 1
EXPECTED: I will pick up the book.
KEY: take-object-from-surface
Observation 2: You pick up the book 1.
THOUGHT: I have the book. I need to find the desklamp.
ACTION: go to desk 1
EXPECTED: I will find a desklamp on or near the desk.
KEY: go-to-lamp-location
Observation 3: On the desk 1, you see a desklamp 1, a pen 2.
THOUGHT: Found the desklamp. Now I can examine the book under it.
ACTION: examine book 1 with desklamp 1
EXPECTED: The book will be examined under the desklamp and the task will complete.
KEY: examine-object-with-lamp
Observation 4: You examine the book 1 with the desklamp 1."""

STAR_EXAMINE_0 = """Task: examine a creditcard with a desklamp.
THOUGHT: I need to find a creditcard. It could be on a shelf, sidetable, or desk.
ACTION: go to shelf 1
EXPECTED: I will find a creditcard on the shelf.
KEY: go-to-shelf-search
Observation 1: On the shelf 1, you see a vase 1 and a statue 1.
THOUGHT: No creditcard on shelf 1. Let me check the sidetable.
ACTION: go to sidetable 1
EXPECTED: I will find a creditcard on the sidetable.
KEY: go-to-sidetable-search
CORRECTION: Creditcards are more likely on sidetables or desks than shelves — check furniture near seating first.
Observation 2: On the sidetable 1, you see a creditcard 1, a book 2.
THOUGHT: Found the creditcard. I need to pick it up then find the desklamp.
ACTION: take creditcard 1 from sidetable 1
EXPECTED: I will pick up the creditcard.
KEY: take-object-from-surface
Observation 3: You pick up the creditcard 1.
THOUGHT: I have the creditcard. I need to find a desklamp.
ACTION: go to desk 1
EXPECTED: There will be a desklamp on the desk.
KEY: go-to-lamp-location
Observation 4: On the desk 1, you see a desklamp 1, a laptop 1.
THOUGHT: Found the desklamp. I can now examine the creditcard.
ACTION: examine creditcard 1 with desklamp 1
EXPECTED: The task will complete after examining.
KEY: examine-object-with-lamp
Observation 5: You examine the creditcard 1 with the desklamp 1."""


# ---------------------------------------------------------------------------
# pick-two-and-place (puttwo)
# ---------------------------------------------------------------------------

STAR_PUTTWO_1 = """Task: put two pillows in sofa.
THOUGHT: I need to find two pillows. They are likely on the sofa or armchair. Let me check the armchair first.
ACTION: go to armchair 1
EXPECTED: I will find a pillow on the armchair.
KEY: go-to-furniture-search
Observation 1: On the armchair 1, you see a pillow 1, a newspaper 1.
THOUGHT: Found a pillow on the armchair. I'll take it.
ACTION: take pillow 1 from armchair 1
EXPECTED: I will pick up the pillow.
KEY: take-object-from-surface
Observation 2: You pick up the pillow 1.
THOUGHT: I have one pillow. I need to find a second one. Let me check the sofa.
ACTION: go to sofa 1
EXPECTED: I will find a second pillow on the sofa.
KEY: go-to-furniture-search
Observation 3: On the sofa 1, you see a pillow 2, a remotecontrol 1.
THOUGHT: I see pillow 2 on the sofa but I already have pillow 1 in hand. I need to place pillow 1 first, then take pillow 2 and put it on the sofa too.
ACTION: put pillow 1 in/on sofa 1
EXPECTED: Pillow 1 will be placed on the sofa.
KEY: put-first-object-in-receptacle
Observation 4: You put the pillow 1 in/on the sofa 1.
THOUGHT: Pillow 1 is on the sofa. But pillow 2 was already there, so both pillows are now on the sofa.
ACTION: put pillow 2 in/on sofa 1
EXPECTED: Pillow 2 is already on the sofa — the task may complete now.
KEY: put-second-object-in-receptacle
Observation 5: You put the pillow 2 in/on the sofa 1."""

STAR_PUTTWO_0 = """Task: put two candles in drawer.
THOUGHT: I need to find two candles. They might be on a shelf, sidetable, or countertop. Let me check the sidetable.
ACTION: go to sidetable 1
EXPECTED: I will find a candle on the sidetable.
KEY: go-to-sidetable-search
Observation 1: On the sidetable 1, you see a candle 1, a book 1.
THOUGHT: Found candle 1. I'll pick it up and then find a second one.
ACTION: take candle 1 from sidetable 1
EXPECTED: I will pick up candle 1.
KEY: take-object-from-surface
Observation 2: You pick up the candle 1.
THOUGHT: I have one candle. I need a second one. Let me check sidetable 2.
ACTION: go to sidetable 2
EXPECTED: I will find a second candle on sidetable 2.
KEY: go-to-sidetable-search
Observation 3: On the sidetable 2, you see a candle 2, a pen 1.
THOUGHT: Found candle 2. But I already have candle 1 in hand. I need to place candle 1 in the drawer, then come back for candle 2.
ACTION: go to drawer 1
EXPECTED: I will arrive at the drawer.
KEY: go-to-target-receptacle
CORRECTION: When carrying one object and finding the second, place the first in the target receptacle before picking up the second.
Observation 4: You arrive at the drawer 1. The drawer 1 is closed.
THOUGHT: Drawer is closed. I need to open it first.
ACTION: open drawer 1
EXPECTED: The drawer will open.
KEY: open-receptacle-before-place
Observation 5: You open the drawer 1.
THOUGHT: Drawer is open. I'll put candle 1 in it.
ACTION: put candle 1 in/on drawer 1
EXPECTED: Candle 1 will be placed in the drawer.
KEY: put-first-object-in-receptacle
Observation 6: You put the candle 1 in/on the drawer 1.
THOUGHT: Now I need to go get candle 2.
ACTION: go to sidetable 2
EXPECTED: I will arrive at sidetable 2 where candle 2 is.
KEY: go-to-second-object-location
Observation 7: On the sidetable 2, you see a candle 2, a pen 1.
THOUGHT: I can take candle 2 now.
ACTION: take candle 2 from sidetable 2
EXPECTED: I will pick up candle 2.
KEY: take-second-object
Observation 8: You pick up the candle 2.
THOUGHT: I have candle 2. Now I need to put it in the drawer.
ACTION: go to drawer 1
EXPECTED: I will arrive at the drawer.
KEY: go-to-target-receptacle
Observation 9: You arrive at the drawer 1. The drawer 1 is open.
THOUGHT: Drawer is already open. I can place candle 2 directly.
ACTION: put candle 2 in/on drawer 1
EXPECTED: The task will be complete.
KEY: put-second-object-in-receptacle
Observation 10: You put the candle 2 in/on the drawer 1."""


# ---------------------------------------------------------------------------
# Task-type → (fewshot_1, fewshot_0) map — mirrors alfworld_3prompts.json structure
# ---------------------------------------------------------------------------

STAR_FEWSHOTS = {
    'put':     (STAR_PUT_1,     STAR_PUT_0),
    'heat':    (STAR_HEAT_1,    STAR_HEAT_0),
    'clean':   (STAR_CLEAN_1,   STAR_CLEAN_0),
    'cool':    (STAR_COOL_1,    STAR_COOL_0),
    'examine': (STAR_EXAMINE_1, STAR_EXAMINE_0),
    'puttwo':  (STAR_PUTTWO_1,  STAR_PUTTWO_0),
}
