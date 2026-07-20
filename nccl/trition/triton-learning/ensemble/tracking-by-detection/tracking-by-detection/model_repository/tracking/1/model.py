import json
import numpy as np
from ocsort.ocsort import OCSort
import triton_python_backend_utils as pb_utils
import cv2
import ast


def filter_detections_outsize_zone(detections, zone):
    filtered_detections = []
    for det in detections:
        center = ((det[0] + det[2]) // 2, det[3])
        if cv2.pointPolygonTest(np.array(zone), center, True) > 0:
            filtered_detections.append(det)
    filtered_detections = np.array(filtered_detections)\
          if len(filtered_detections) > 0 else np.empty((0, 6))
    return filtered_detections


class TritonPythonModel:
    def initialize(self, args):
        print('Tracking: initialization...')
        model_config = json.loads(args['model_config'])
        tracks = pb_utils.get_output_config_by_name(
            model_config, 'tracks'
        )
        self.tracks_dtype = pb_utils.triton_string_to_numpy(
            tracks['data_type']
        )
        self.trackers = {}
        self.zones = {}

    def execute(self, requests):
        responses = []
        for request in requests:
            seq_start = pb_utils.get_input_tensor_by_name(request, "START")
            seq_start = seq_start.as_numpy()[0][0]
            seq_id = pb_utils.get_input_tensor_by_name(request, "CORRID")
            seq_id = seq_id.as_numpy()[0][0]
            seq_end = pb_utils.get_input_tensor_by_name(request, "END")
            seq_end = seq_end.as_numpy()[0][0]

            parameters = ast.literal_eval(request.parameters())
            
            if 'zone' in parameters:
                zone = list(ast.literal_eval(parameters['zone']))
                if len(zone) > 2:
                    self.zones[seq_id] = zone
                else:
                    self.zones.pop(seq_id, None)

            if seq_start:
                self.trackers[seq_id] = OCSort(
                    per_class=True,
                    det_thresh=0,
                    max_age=30,
                    min_hits=1,
                    asso_threshold=0.3,
                    delta_t=3,
                    asso_func='giou',
                    inertia=0.2,
                    use_byte=False,
                )
            if seq_end:
                self.trackers.pop(seq_id)

            detections = pb_utils.get_input_tensor_by_name(request, 'detections')
            frame = pb_utils.get_input_tensor_by_name(request, 'frame')
            detections = np.squeeze(detections.as_numpy(), axis=0)

            if seq_id in self.zones:
                zone = self.zones[seq_id]
                detections = filter_detections_outsize_zone(detections, zone)
            frame = np.squeeze(frame.as_numpy(), axis=0)

            tracks = self.trackers[seq_id].update(detections, frame)
            tracks = np.expand_dims(tracks, axis=0)
            tracks = pb_utils.Tensor(
                'tracks', tracks.astype(self.tracks_dtype)
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[tracks]
            )
            responses.append(inference_response)
        return responses

    def finalize(self):
        print('Tracking: cleaning up...')
