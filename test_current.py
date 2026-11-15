"""Small synthetic checks of state ordering, profile shape, and score fusion."""
import numpy as np
from current.direct_entity_state import EntityHistory
from current.history_features import views
from current.score_fusion import agreement

state=EntityHistory()
write={"timestamp_ns":1,"action":"file_write","actor":"p","target":"f",
       "account":None,"host":"h","outcome":"success"}
read={"timestamp_ns":2,"action":"file_read","actor":"p","target":"f",
      "account":None,"host":"h","outcome":"success"}
assert not state.before(write).any()
state.apply(write)
before=state.before(read)
assert before[27+6]==1, "prior successful write by same actor is missing"
assert before.shape==(39,)
state.apply(read)
old=np.array([.5,.5])
host=np.array([.6,.6])
direct=np.array([.8,.2])
got=agreement(old,host,direct,np.array([True,True]),np.array([True,True]))
np.testing.assert_allclose(got,[.625,.52])
a=np.zeros((1,935))
out=views(a,a)
assert out["host_profile"].shape==(1,935)
assert out["entity_only_profile"].shape==(1,935)
assert out["profile_contrast"].shape==(1,1059)
print("Synthetic checks passed: event-before-write, role link, score fusion, input widths.")

