# Third-party model notices

The application downloads unmodified model weights from OpenCV Zoo at commit
`47534e27c9851bb1128ccc0102f1145e27f23f98`.

YuNet is covered by `YuNet-MIT.txt`. The facial-expression model directory states
that its files are covered by Apache 2.0; the OpenCV Zoo root license is included
as `OpenCV-Zoo-Apache-2.0.txt` (that model directory has no separate LICENSE file
at the pinned revision).

The expression alignment and preprocessing follow OpenCV Zoo's
`models/facial_expression_recognition/facial_fer_model.py`:

Copyright (C) 2022, Shenzhen Institute of Artificial Intelligence and Robotics for
Society, all rights reserved. Third party copyrights are property of their
respective owners.

The application rewrites the alignment implementation with a direct NumPy
least-squares solve and retains the model's reference landmark coordinates,
normalization, and output class order. Progressive Teacher was contributed to
OpenCV Zoo by Jing Jiang; the MobileFaceNet ONNX implementation is credited there
to Chengrui Wang. See the project README for source links.
