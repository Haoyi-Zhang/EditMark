"""Run fixed upstream in-process tests against the exact supplied module.

This is a bounded trusted-source experiment harness, not a security sandbox.
Child-command, C-accelerator-only and documentation tests are outside its
declared in-process Python contract; all such exclusions are recorded by ID.
"""
from __future__ import annotations
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
ACTIVE=False

def flatten(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite):yield from flatten(item)
        else:yield item

def exclusion(test,module):
    cls=type(test).__name__
    if cls.endswith('C'):return 'C-accelerator-only class'
    if cls in ('DocTestCase','DocTests'):return 'documentation-only tests'
    if module=='calendar' and cls=='CommandLineTestCase':return 'subprocess CLI does not share supplied-module provenance'
    return None

class RecordedResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.outcomes=[];self.started=[]
    def startTest(self,test):
        global ACTIVE
        super().startTest(test);self.started.append(test.id());ACTIVE=True
    def stopTest(self,test):
        global ACTIVE
        ACTIVE=False;super().stopTest(test)
    def addSubTest(self,test,subtest,err):
        super().addSubTest(test,subtest,err)
        if err is not None:self.outcomes.append(dict(id=test.id(),status='subtest_failure',detail=str(subtest)))
    def addSuccess(self,test):super().addSuccess(test);self.outcomes.append(dict(id=test.id(),status='passed'))
    def addFailure(self,test,err):super().addFailure(test,err);self.outcomes.append(dict(id=test.id(),status='failed'))
    def addError(self,test,err):super().addError(test,err);self.outcomes.append(dict(id=test.id(),status='error'))
    def addSkip(self,test,reason):super().addSkip(test,reason);self.outcomes.append(dict(id=test.id(),status='skipped',reason=reason))
    def addExpectedFailure(self,test,err):super().addExpectedFailure(test,err);self.outcomes.append(dict(id=test.id(),status='expected_failure'))
    def addUnexpectedSuccess(self,test):super().addUnexpectedSuccess(test);self.outcomes.append(dict(id=test.id(),status='unexpected_success'))

def run(module,candidate,test_root):
    resource.setrlimit(resource.RLIMIT_CPU,(20,20))
    resource.setrlimit(resource.RLIMIT_AS,(1024*1024*1024,1024*1024*1024))
    resource.setrlimit(resource.RLIMIT_FSIZE,(32*1024*1024,32*1024*1024))
    candidate=candidate.resolve();test_root=test_root.resolve()
    expected=candidate/(module+'.py')
    source=expected.read_bytes();random.seed(0)
    sys.path[:0]=[str(candidate),str(test_root)]
    sys.modules.pop(module,None)
    lines=set();calls=set();loaded=[]
    def trace(frame,event,arg):
        if ACTIVE and frame.f_code.co_filename==str(expected):
            if event=='line':lines.add(frame.f_lineno)
            elif event=='call':calls.add(frame.f_code.co_name)
        return trace
    log=io.StringIO();stdout=io.StringIO();report=dict(module=module,
        source_sha256=hashlib.sha256(source).hexdigest(),status='unavailable')
    try:
        with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(log):
            sys.settrace(trace)
            unit=importlib.import_module('test.test_'+module)
            whole=unittest.defaultTestLoader.loadTestsFromModule(unit)
            selected=[];excluded=[]
            for case in flatten(whole):
                reason=exclusion(case,module)
                if reason:excluded.append(dict(id=case.id(),reason=reason))
                else:selected.append(case)
            # All test-created Python copies must come from the candidate path,
            # including import_fresh_module's pure-Python accelerator variants.
            for name,value in vars(unit).items():
                if getattr(value,'__name__',None)==module and hasattr(value,'__file__'):
                    actual=Path(value.__file__).resolve()
                    if actual!=expected:raise ValueError('upstream test used an unbound module copy')
                    loaded.append(dict(binding=name,source_sha256=hashlib.sha256(actual.read_bytes()).hexdigest()))
            value=sys.modules.get(module)
            if value is not None and Path(value.__file__).resolve()==expected:
                loaded.append(dict(binding='sys.modules',source_sha256=hashlib.sha256(expected.read_bytes()).hexdigest()))
            if not selected or not loaded:raise ValueError('No bound upstream Python tests')
            suite=unittest.TestSuite(selected)
            result=unittest.TextTestRunner(stream=log,verbosity=1,resultclass=RecordedResult).run(suite)
            report.update(status='completed',tests_scheduled=len(selected),tests_run=result.testsRun,
                test_ids=[test.id() for test in selected],excluded_tests=excluded,
                outcomes=result.outcomes,started_tests=result.started,failures=len(result.failures),errors=len(result.errors),
                skipped=len(result.skipped),expected_failures=len(result.expectedFailures),
                unexpected_successes=len(result.unexpectedSuccesses),
                passed=result.wasSuccessful() and result.testsRun>len(result.skipped) and bool(lines),bound_modules=loaded)
    except BaseException as error:
        report.update(error=type(error).__name__,detail=str(error),passed=False)
    finally:sys.settrace(None)
    report.update(executed_lines=sorted(lines),executed_functions=sorted(calls),
        captured_log=log.getvalue(),captured_stdout=stdout.getvalue())
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--module',required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--test-root',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.module,a.candidate,a.test_root),ensure_ascii=True,allow_nan=False))
