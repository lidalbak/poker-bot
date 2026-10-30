"""Position must survive folds; blind defense must respect price and action."""
import itertools
import unittest
from poker.protocol import parse, validate
from poker.strategy import action_order, live_action_order, seats_behind, decide
from tests.scenarios import base

class PreflopRegression(unittest.TestCase):
    def test_folds_preserve_order(self):
        for n in range(2, 6):
            for button in range(n):
                ids = ['seat-%s' % (i+1) for i in range(n)]
                expected = action_order(ids, ids[button], 'preflop')
                for mask in itertools.product([False, True], repeat=n):
                    p = base('preflop', [], ['As','Kd'], 4900, 100, 100, ['fold','check','raise'])
                    p['state']['players'] = p['state']['players'][:n]
                    p['state']['hand']['buttonId'] = ids[button]
                    for player, folded in zip(p['state']['players'],mask):
                        player['folded'] = folded
                    s=parse(p)
                    want=[pid for pid in expected if not mask[ids.index(pid)]]
                    self.assertEqual(live_action_order(s),want)
                    if s.you_id in want:
                        self.assertEqual(seats_behind(s),len(want)-1-want.index(s.you_id))

    def spot(self, amount=200, hole=None):
        p=base('preflop', [], hole or ['7s','6s'], 4900,100,amount,['fold','call','raise'])
        p['state']['hand'].update(buttonId='seat-4',pot=amount+150,ante=0)
        for q in p['state']['players']:
            q['folded']=q['id'] not in ('seat-1','seat-4')
        p['state']['players'][3]['bet']=amount
        return parse(p)

    def test_defend_small_button_open(self):
        s=self.spot()
        self.assertEqual(decide(s),('call',None))
        self.assertEqual(validate(s.legal,*decide(s)),decide(s))

    def test_no_loose_defense_against_large_raise(self):
        self.assertEqual(decide(self.spot(1000)),('fold',None))

    def test_trash_still_folds(self):
        self.assertEqual(decide(self.spot(300,['7s','2d'])),('fold',None))

if __name__=='__main__': unittest.main()
