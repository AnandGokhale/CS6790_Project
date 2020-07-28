# RT-SVO
Project on visual odometry For The Geometry and Photometry Course offered in IIT Madras, Spring 2020.

The code provided is a python implementation of [[1]](#1).

## Instructions to run
To evaluate on a sequence, run 
<pre><code>python svo.py --seqPath /path/to/kitti/sequence/ --feature feature_detector --resultsPath /path/to/results/file
</code></pre>
The <code> feature_detector </code> can be one of "FAST" or "BRIEF". Example  usage : 
<pre><code>python svo.py --seqPath ./dataset/sequences/00/ --feature FAST --resultsPath ./results_00.txt
</code></pre>


## References 
<a id="1">[1]</a> 
Andrew Howard. 
Real-time stereo visual odometry for autonomous ground vehicles.
In *2008 IEEE/RSJ International Conference on Intelligent Robots and Systems*, pages 3946–3952.IEEE, 2008.
