"""Temporal-access, persistence and label-independence tests on explicit streams."""
from pathlib import Path
import copy,sys,unittest
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from portable.replay import replay

def observation(t,action='file_write',sid='a',close=False):
    event=dict(event_id=str(t),timestamp_ns=t,action=action,actor='process',target='file',account='user',
               host='host',outcome='success',focus=['process','file','user','host'],
               entity_observations=[{'key':k} for k in ['process','file','user','host']])
    return dict(session_id=sid,event=event,close_session=close)

class TestReplay(unittest.TestCase):
    def test_first_read_and_persistent_write_fact(self):
        a,_=replay([observation(1,close=True),observation(2,'file_read','b',True)])
        np.testing.assert_array_equal(a['host124'][0],0)
        self.assertTrue(np.any(a['host124'][1]>0))
        self.assertEqual(float(a['links24'][1][6]),1.)
        self.assertEqual(float(a['links24'][1][18]),1.)
    def test_future_append_cannot_change_closed_prefix(self):
        base=[observation(1,close=True)]
        x,_=replay(copy.deepcopy(base));y,_=replay(base+[observation(2,'exec','b',True)])
        for k in ['host124','entity124','roles54','links24']:np.testing.assert_array_equal(x[k][0],y[k][0])
    def test_unlabeled_observations_still_update(self):
        a=observation(1,sid=None);b=observation(2,'file_read','b',True)
        x,_=replay([a,b]);self.assertTrue(np.any(x['host124'][0]>0))
    def test_labels_do_not_affect_replay(self):
        rows=[observation(1),observation(2,'file_read','b',True)]
        changed=copy.deepcopy(rows)
        for row in changed:row['attack_label']=1;row['event']['y']=1
        a,_=replay(rows);b,_=replay(changed)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
    def test_nonchronological_rejected(self):
        with self.assertRaises(ValueError):replay([observation(3),observation(1)])
    def test_reopening_closed_session_rejected(self):
        with self.assertRaises(ValueError):replay([observation(1,close=True),observation(2)])
if __name__=='__main__':unittest.main()
