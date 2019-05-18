from kubernetes import client

from fairing.preprocessors import converted_notebook
import json
import nbconvert
from nbconvert.preprocessors import Preprocessor as NbPreProcessor
import numpy as np
from pathlib import Path
import re
import requests

class FilterIncludeCell(NbPreProcessor):
    """Notebook preprocessor that only includes cells that have a comment fairing:include-cell"""
    _pattern = re.compile('.*fairing:include-cell.*')

    def filter_include_cell(self, src):
        filtered = []
        for line in src.splitlines():
            match = self._pattern.match(line)
            if match:
                return src
                filtered.append(line)
        return ''

    def preprocess_cell(self, cell, resources, index):
        if cell['cell_type'] == 'code':
            cell['source'] = self.filter_include_cell(cell['source'])
        return cell, resources   


class ConvertNotebookPreprocessorWithFire(converted_notebook.ConvertNotebookPreprocessor):
    """Create an entrpoint using pyfire."""
    def __init__(self, class_name = None, *args, **kwargs):

        if not "notebook_preprocessor" in kwargs:
            kwargs["notebook_preprocessor"] = FilterIncludeCell()
        super().__init__(*args, **kwargs)

        self.class_name=class_name

    def preprocess(self):
        exporter = nbconvert.PythonExporter()
        exporter.register_preprocessor(self.notebook_preprocessor, enabled=True)
        contents, _ = exporter.from_filename(self.notebook_file)
        converted_notebook = Path(self.notebook_file).with_suffix('.py')
        with open(converted_notebook, 'w') as f:
            f.write(contents)
            f.write("\n")
            f.write("""
if __name__ == "__main__":
  import fire
  import logging
  logging.basicConfig(format='%(message)s')
  logging.getLogger().setLevel(logging.INFO)
  fire.Fire({0})
""".format(self.class_name))
        self.executable = converted_notebook
        results =  [converted_notebook]
        results.extend(self.input_files)
        return results

def has_volume(pod_spec, pvc_name):
    if not pod_spec.containers[0].volumes:
        return False
    
    for v in pod_spec.containers[0].volumes:
        if v.name == pvc_name:
            return True
    
    return False

def add_pvc_mutator(pvc_name, mount_path):
    """Generate a pod mutator to add a pvc."""
    
    def add_pvc(kube_manager, pod_spec, namespace):
        """Add a pvc to the specified pod spec."""
        volume_mount = client.V1VolumeMount(
            name=pvc_name, mount_path=mount_path, read_only=False)


        if not pod_spec.containers[0].volume_mounts:
            pod_spec.containers[0].volume_mounts = []

        pod_spec.containers[0].volume_mounts.append(volume_mount)

        volume = client.V1Volume(
            name=pvc_name,
            persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(pvc_name))
        if pod_spec.volumes:
            pod_spec.volumes.append(volume)
        else:
            pod_spec.volumes = [volume]

    return add_pvc

def predict_nparray(url, data, feature_names=None):
    pdata={
        "data": {
            "names":feature_names,
            "tensor": {
                "shape": np.asarray(data.shape).tolist(),
                "values": data.flatten().tolist(),
            },
        }
    }
    serialized_data = json.dumps(pdata)
    r = requests.post(url, data={'json':serialized_data})
    return r