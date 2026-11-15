"""Isolation fixtures against the actual existing Environment implementation."""
import argparse,copy,importlib.util,json,pathlib,sys
sys.dont_write_bytecode=True
from history_ledger import CommonLedger,MemoryStore

def main():
    p=argparse.ArgumentParser();p.add_argument('--engine',default='<RESEARCH_ROOT>/reproductions/ait_ads_session_state_20261023/engine.py');a=p.parse_args()
    spec=importlib.util.spec_from_file_location('ledger_fixture_engine',a.engine);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    Environment=m.Environment
    def event(i,t,attrs=None):
        return {'event_id':f'{i:032x}','timestamp_ns':int(t*10**9),'focus':['host:h'],
                'entity_observations':[{'key':'host:h','kind':'host','attributes':attrs or {}}],
                'actor':None,'target':None,'host':'host:h','action':'unknown','outcome':'unknown'}
    store=MemoryStore();env=Environment(store);ledger=CommonLedger(Environment)
    # An old path remains in native lifetime state, but expires from the common
    # ledger and must not enter either text or numeric H inputs.
    events=[event(1,0,{'path':'/outside/window/secret.biz'})]+[event(i,50000+i) for i in range(2,10)]
    for e in events:
        before=env.before(e);ledger.before(before,e);env.apply(e);ledger.observe(e,'s',1,2)
    current=event(10,50010);before=env.before(current);unchanged=copy.deepcopy(before)
    bounded,prior=ledger.before(before,current)
    assert before==unchanged,'Ledger mutated native real history'
    assert before['entities'][0]['events_observed']==9
    assert bounded['entities'][0]['events_observed']==6
    assert bounded['entities'][0]['attributes'].get('path') is None
    assert len(prior)==6 and current['event_id'] not in prior
    assert all(current['timestamp_ns']-x['event']['timestamp_ns']<=43200*10**9 for x in prior.values())
    # Equal timestamps admit stable earlier IDs and reject future/current refs.
    same=event(11,50010);env.apply(current);ledger.observe(current,'s',1,2)
    _,prior=ledger.before(env.before(same),same)
    assert current['event_id'] in prior
    invalid=copy.deepcopy(env.before(same));invalid['entities'][0]['recent'].append({'event_id':same['event_id'],'timestamp_ns':same['timestamp_ns']})
    try:ledger.before(invalid,same)
    except AssertionError:pass
    else:raise AssertionError('Current-event contamination not rejected')
    print(json.dumps({'passed':True,'checks':['lifetime path excluded','at most six records/entity','same before entity selector','real native state not mutated','past-only and stable equal-time order','current-event contamination rejected'],'ledger_receipt':ledger.receipt()},indent=2))
if __name__=='__main__':main()
