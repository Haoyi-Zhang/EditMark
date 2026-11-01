"""Same upstream assertions with function-entry provenance instead of line tracing.

This separately registered validator avoids statement-tracing overhead. It does
not claim statement coverage and does not replace the original activity record.
"""
import argparse
import contextlib
import hashlib
import importlib
import io
import json
from pathlib import Path
import random
import resource
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import upstream_suite as base

def run(module,candidate,test_root):
    resource.setrlimit(resource.RLIMIT_CPU,(20,20));resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    candidate=candidate.resolve();expected=candidate/(module+'.py');digest=hashlib.sha256(expected.read_bytes()).hexdigest()
    sys.path[:0]=[str(candidate),str(test_root.resolve())];sys.modules.pop(module,None);random.seed(0)
    entered=set();bound=[];log=io.StringIO();stdout=io.StringIO()
    def profile(frame,event,arg):
        if base.ACTIVE and event=='call' and frame.f_code.co_filename==str(expected):entered.add(frame.f_code.co_name)
    report=dict(module=module,source_sha256=digest,status='unavailable',passed=False,provenance_kind='function-entry calls; not statement coverage')
    try:
        with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(log):
            unit=importlib.import_module('test.test_'+module)
            selected=[];excluded=[]
            for test in base.flatten(unittest.defaultTestLoader.loadTestsFromModule(unit)):
                reason=base.exclusion(test,module)
                if reason:excluded.append(dict(id=test.id(),reason=reason))
                else:selected.append(test)
            for name,value in vars(unit).items():
                if getattr(value,'__name__',None)==module and hasattr(value,'__file__'):
                    p=Path(value.__file__).resolve()
                    if p!=expected:raise ValueError('Unbound upstream module')
                    bound.append(dict(binding=name,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
            value=sys.modules.get(module)
            if value is not None and Path(value.__file__).resolve()==expected:bound.append(dict(binding='sys.modules',source_sha256=digest))
            if not selected or not bound:raise ValueError('No bound upstream tests')
            sys.setprofile(profile)
            result=unittest.TextTestRunner(stream=log,verbosity=1,resultclass=base.RecordedResult).run(unittest.TestSuite(selected))
            sys.setprofile(None)
            report.update(status='completed',tests_scheduled=len(selected),tests_run=result.testsRun,
                test_ids=[test.id() for test in selected],excluded_tests=excluded,outcomes=result.outcomes,started_tests=result.started,
                failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
                expected_failures=len(result.expectedFailures),unexpected_successes=len(result.unexpectedSuccesses),
                passed=result.wasSuccessful() and result.testsRun>len(result.skipped) and bool(entered),bound_modules=bound)
    except BaseException as error:report.update(error=type(error).__name__,detail=str(error))
    finally:sys.setprofile(None)
    report.update(entered_python_functions=sorted(entered),captured_log=log.getvalue(),captured_stdout=stdout.getvalue())
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--module',required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--test-root',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.module,a.candidate,a.test_root),allow_nan=False))
