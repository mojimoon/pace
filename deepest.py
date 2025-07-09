import os.path
import sys
import random
import numpy as np

class ActiveSet:
    def __init__(self, N, T, occurrenceP):
        self.testFrame = []
        self.id = [0] * N
        self.outcome = [False] * N
        self.weights = [0.0] * T
        self.occurrenceProb = list(occurrenceP)
        self.outcomeSum = 0
        self.outcomeSumX = 0.0
        self.qi = 0.0

    def getWeights(self, k): return self.weights[k]

    def update(self, tf_name, tf_index, passed, upweights):
        idx = len(self.testFrame)
        self.testFrame.append(tf_name)
        self.id[idx] = tf_index
        self.outcome[idx] = passed

        self.weights = [
            w + upweights[tf_index][i] if i not in self.id else 0.0
            for i, w in enumerate(self.weights)
        ]

        if passed:
            self.outcomeSum += 1
            self.outcomeSumX += self.occurrenceProb[tf_index]

    def testFrameExtraction(self, d):
        total = sum(self.weights)
        if total == 0:
            self.qi = 1 / (len(self.weights) - len(self.testFrame))
            return 0

        probs = [w / total for w in self.weights]
        selected = np.random.choice(len(self.weights), p=probs)
        self.qi = d * probs[selected] + (1 - d) / (len(self.weights) - len(self.testFrame))
        return selected

    def qiCalculation(self, d, k):
        total = sum(self.weights)
        self.qi = (
            d * (self.weights[k] / total) + (1 - d) / (len(self.weights) - len(self.testFrame))
            if total > 0 else
            1 / (len(self.weights) - len(self.testFrame))
        )


class TestFrame:
    def __init__(self, name, tfID, failureProb, occurrenceProb, output, fail):
        self.name = name
        self.tfID = tfID
        self.failureProb = failureProb
        self.occurrenceProb = occurrenceProb
        self.output = output
        self.fail = fail

    def extractAndExecuteTestCase(self):
        # Returns True if this test frame simulates a failure
        return self.fail

    def getOutput(self):
        return self.output

    def getName(self):
        return self.name

    def setName(self, name):
        self.name = name

    def getTfID(self):
        return self.tfID

    def setTfID(self, tfID):
        self.tfID = tfID

    def getFailureProb(self):
        return self.failureProb

    def setFailureProb(self, failureProb):
        self.failureProb = failureProb

    def getOccurrenceProb(self):
        return self.occurrenceProb

    def setOccurrenceProb(self, occurrenceProb):
        self.occurrenceProb = occurrenceProb
class DeepESTSelector:
    def __init__(self):
        self.numfp = 0
        self.z = []
        self.failedRequest = []

    def getZ(self):
        return self.z

    def getnumfp(self):
        return self.numfp

    def getfailedRequest(self):
        return self.failedRequest

    def selectAndRunTestCase(self, n, testFrameList, weightsMatrix, d):
        self.failedRequest.clear()
        self.numfp = 0

        if n > len(testFrameList) or n <= 0:
            return [-1.0, -1.0]
        if d <= 0 or d >= 1:
            return [-2.0, -2.0]

        scompl = list(range(len(testFrameList)))
        occurrenceProb = [tf.getOccurrenceProb() for tf in testFrameList]
        randomNum = random.randint(0, len(testFrameList) - 1)
        ak = ActiveSet(n, len(testFrameList), occurrenceProb)
        name = testFrameList[randomNum].getTfID()
        esito = testFrameList[randomNum].extractAndExecuteTestCase()
        tc = testFrameList[randomNum].getName()
        ak.activeSetUpdate(name, randomNum, esito, weightsMatrix)
        scompl.remove(randomNum)

        if esito:
            y = 1
            self.numfp += 1
            self.failedRequest.append(tc)
        else:
            y = 0

        estimationX = [0] * n
        estimationX[0] = len(testFrameList) * (occurrenceProb[randomNum] * y)

        k = 1
        while k < n:
            weightsSum = sum(ak.getWeights(i) for i in scompl)
            if weightsSum == 0:
                prob = d + 0.1
            else:
                prob = random.random()

            if prob <= d: # WBS
                current_tf = ak.testFrameExtraction(d)
            else: # random selection SRS
                random_idx = random.randint(0, len(scompl) - 1)
                current_tf = scompl[random_idx]

            name = testFrameList[current_tf].getTfID()
            esito = testFrameList[current_tf].extractAndExecuteTestCase()
            tc = testFrameList[current_tf].getName()

            ziX = ak.getOutcomeSumX()

            if esito:
                if prob > d:
                    ak.qiCalculation(d, current_tf)
                ziX += occurrenceProb[current_tf] / ak.qi
                self.numfp += 1
                self.failedRequest.append(tc)

            ak.activeSetUpdate(name, current_tf, esito, weightsMatrix)

            if prob <= d:
                scompl.remove(current_tf)
            else:
                scompl.pop(random_idx)

            estimationX[k] = ziX
            k += 1
        return ak.testFrame
        #self.z = estimationX
        #return self.estimatorBoCSP(n, estimationX)

    def estimatorBoCSP(self, n, estimationX):
        sumX = sum(estimationX)
        mean = sumX / n
        variance = sum((estimationX[i] - estimationX[0]) ** 2 for i in range(1, n)) / (n * (n - 1))
        return [mean, variance]
import subprocess
import tempfile
from pathlib import Path
from typing import List, Any


class TestCase:
    string_ok = ""
    string_not_ok = "different results"

    class ExecutionState:
        EXECUTED = "executed"
        NOT_EXECUTED = "notExecuted"

    def __init__(self, name: str, tcID: str, root_directory: Path = None):
        self.name = name
        self.tcID = tcID
        self.outcome = False
        self.execution_state = self.ExecutionState.NOT_EXECUTED
        self.path_root_directory = root_directory or Path.cwd()
        self.number_of_commands = 0
        self.list_of_commands: List[str] = []
        self.inputs: List[Any] = []
        self.max_number_of_inputs = 0
        self.expected_occurrence_probability = 0.0
        self.real_occurrence_probability = 0.0
        self.expected_failure_likelihood = 0.0

    def __eq__(self, other):
        if not isinstance(other, TestCase):
            return False
        return self.tcID == other.tcID

    def __hash__(self):
        return hash(self.name)

    def run_test_case_dummy(self, string: str) -> bool:
        """Simulates execution."""
        self.outcome = self.get_outcome()
        self.execution_state = self.ExecutionState.EXECUTED
        return self.outcome

    def run_test_case(self) -> bool:
        """Actually runs the shell script and checks output."""
        script_path = self._create_temp_script()

        print(f"\n\n***** Running Test {self.name}. ******\nExecuted Temporary Script: {script_path}")
        try:
            result = subprocess.run(
                [script_path],
                cwd=self.path_root_directory,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            print(f"\n'Standard Output' Printed by the test case execution {self.name}: {stdout}")
            if result.returncode != 0:
                print(f"\n'Standard Error' Printed by the test case execution {self.name}: {stderr}")
                print(f"Execution State: {result.returncode}")
                self.execution_state = self.ExecutionState.NOT_EXECUTED
            else:
                self.execution_state = self.ExecutionState.EXECUTED

            if stdout == self.string_not_ok:
                self.outcome = False
            elif stdout == self.string_ok:
                self.outcome = True

        finally:
            script_path.unlink()  # delete the temp script

        print(f"\n ** Test {self.name} terminated **\n")
        return self.outcome

    def _create_temp_script(self) -> Path:
        """Create a temporary bash script with the list of commands."""
        fd, path = tempfile.mkstemp(suffix=".sh", dir=self.path_root_directory)
        script_path = Path(path)
        script_path.chmod(0o755)

        with open(script_path, 'w', encoding='utf-8') as f:
            f.write("#!/bin/bash\n")
            for cmd in self.list_of_commands:
                f.write(cmd + "\n")

        return script_path

    # --- Getter / Setter properties ---

    def get_name(self):
        return self.name

    def set_name(self, name: str):
        self.name = name

    def get_number(self):
        return self.tcID

    def set_number(self, number: str):
        self.tcID = number

    def get_number_of_commands(self):
        return self.number_of_commands

    def set_number_of_commands(self, num: int):
        self.number_of_commands = num

    def get_list_of_commands(self):
        return self.list_of_commands

    def set_list_of_commands(self, cmds: List[str]):
        self.list_of_commands = cmds

    def get_expected_occurrence_probability(self):
        return self.expected_occurrence_probability

    def set_expected_occurrence_probability(self, value: float):
        self.expected_occurrence_probability = value

    def get_real_occurrence_probability(self):
        return self.real_occurrence_probability

    def set_real_occurrence_probability(self, value: float):
        self.real_occurrence_probability = value

    def set_outcome(self, outcome: bool):
        self.outcome = outcome

    def get_outcome(self) -> bool:
        return self.outcome

    def get_inputs(self):
        return self.inputs

    def set_inputs(self, inputs: List[Any]):
        self.inputs = inputs

    def get_max_number_of_inputs(self):
        return self.max_number_of_inputs

    def set_max_number_of_inputs(self, value: int):
        self.max_number_of_inputs = value

    def get_tcID(self):
        return self.tcID

    def set_tcID(self, tcID: str):
        self.tcID = tcID

    def get_execution_state(self):
        return self.execution_state

    def set_execution_state(self, state: str):
        self.execution_state = state
from typing import List

class InitializerTF:
    def __init__(self, path: str):
        self.csv_file = path

    def readTestFrames(self, key: int, size: int) -> List[TestFrame]:
        test_frames = []
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                header = next(f)  # Skip header
                occ = 1.0 / size  # Uniform occurrence probability

                for i, line in enumerate(f):
                    parts = line.strip().split(",")
                    outcome = parts[1]
                    fail = outcome != "Pass"
                    val = float(parts[key])

                    if key in [3, 6]:  # invert confidence and combo
                        val = 1.0 - val
                        # optional epsilon:
                        # if val == 0.0:
                        #     val = 1e-9

                    tf = TestFrame(
                        name=str(len(test_frames)),
                        tfID=str(len(test_frames)),
                        failureProb=val,
                        occurrenceProb=occ,
                        output=parts[2],
                        fail=fail
                    )
                    test_frames.append(tf)

                if i + 1 != size:
                    print("[WARNING] The size is lower/greater!!!")

        except FileNotFoundError as e:
            print(f"[ERROR] File not found: {e}")
        except Exception as e:
            print(f"[ERROR] {e}")

        return test_frames

    def weightedMatrixComputation_threshold(self, tf: List[TestFrame], key: int, threshold: float) -> List[List[float]]:
        size = len(tf)
        wm = [[0.0 for _ in range(size)] for _ in range(size)]

        for i in range(size):
            for j in range(size):
                conf_j = tf[j].getFailureProb()
                if conf_j > threshold:
                    wm[i][j] = conf_j
                else:
                    wm[i][j] = 0.0

        return wm

def main(dataset, model_name, budget, aux_variable='combo', threshold=0.7):
    dataset_path = f'./AllResult/DeepEST/{model_name}_{dataset}.csv'
    # Determine feature key and adjust threshold
    if aux_variable == "confidence":
        key = 3
        threshold = 1 - threshold
    elif aux_variable == "dsa":
        key = 4
    elif aux_variable == "lsa":
        key = 5
    else:  # combo or others
        key = 6
        threshold = 1 - threshold

    print(f"Approach execution on {dataset_path} with auxiliary variable {aux_variable} and budget {budget}")

    rep = 1 #30
    csv_reader = InitializerTF(dataset_path)
    test_frames = csv_reader.readTestFrames(key, 10000)

    aws = DeepESTSelector()
    weights_matrix = csv_reader.weightedMatrixComputation_threshold(test_frames, key, threshold)
    breakpoint()
    rel_arr = []
    num_fp = []
    #for i in range(rep):
        #rel_arr.append(1 - rel[0])
        #num_fp.append(aws.getnumfp())

    selected_test_suite = aws.selectAndRunTestCase(budget, test_frames, weights_matrix, 0.8)
    return np.array(selected_test_suite).astype(np.int32)
    #for i in range(rep):
    #    print(f"Repetition {i+1}) Estimated Accuracy: {rel_arr[i]:.4f} | Number of failed tests: {num_fp[i]}")

def save_csv():
    import numpy as np
    import pandas as pd
    path = '/home/jzhang2297/empirical/pace/AllResult/DSA/'
    # Load DSA data
    dataset = 'mnist_label'
    models = ['lenet5']
    for model in models:
        if os.path.exists(f'{model}_{dataset}.csv'):
            continue
        dsa = np.load(path + f'{dataset}_{model}_dsa.npy')  # should be shape (10000,)

        # Check length
        assert len(dsa) == 10000, "DSA must have shape (10000,)"

        # Normalize DSA to 0-1 range
        norm_dsa = (dsa - np.min(dsa)) / (np.max(dsa) - np.min(dsa))

        conf = np.load(path + f'{dataset}_{model}_confidence.npy')
        confidence = np.max(conf, axis=1)
        # Compute conf * (1 - norm_dsa)
        combo = confidence * (1 - norm_dsa)

        # Build dataframe
        df = pd.DataFrame({
            'ID': np.arange(10000),
            'Outcome': ['Pass'] * 10000,
            'SUT': [0] * 10000,
            'Confidence': confidence,
            'dsa': dsa,
            'lsa': [0] * 10000,
            'conf*(1-norm_dsa)': combo
        })

        # Save to CSV
        df.to_csv(f'{model}_{dataset}.csv', index=False)
        print(f"CSV file '{model}_{dataset}.csv' created successfully.")

if __name__ == "__main__":
    #save_csv()

    selectedX = main(sys.argv[1:]) # python3 deepest.py ./AllResult/DeepEST/lenet1_mnist.csv combo 0.7 50
    selectedy = testy[selectedX]