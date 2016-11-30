
import copy
import numpy as np

from matplotlib import pyplot as plt
from sklearn import manifold, random_projection

plt.ion()
plt.close('all')

"""
The demo script uses multi-dimensional scaling (MDS) to visualize SDR clusters

The MDS algorithm is a widely used information visualization technique that
takes pair-wise distance between a set of high-dimensional data points as input.
The algorithm places these data points in a low-dimensional space such that
the distance is preserved as well as possible.

We also compare MDS with a naive random projection algorithm. The random
projection algorithm projects a N-dimensional SDR to a 2D space with a random
N x 2 matrix. The semantic distance is preserved to some extend when noise
is low, but the performance is much worse than the MDS algorithm
"""

def percentOverlap(x1, x2):
  """
  Computes the percentage of overlap between vectors x1 and x2.

  @param x1   (array) binary vector
  @param x2   (array) binary vector
  @param size (int)   length of binary vectors

  @return percentOverlap (float) percentage overlap between x1 and x2
  """
  nonZeroX1 = np.count_nonzero(x1)
  nonZeroX2 = np.count_nonzero(x2)
  minX1X2 = min(nonZeroX1, nonZeroX2)
  percentOverlap = 0
  if minX1X2 > 0:
    percentOverlap = float(np.dot(x1.T, x2)) / float(minX1X2)
  return percentOverlap


def generateSDR(n, w):
  """
  Generate a random n-dimensional SDR with w bits active
  """
  sdr = np.zeros((n, ))
  randomOrder = np.random.permutation(np.arange(n))
  activeBits = randomOrder[:w]
  sdr[activeBits] = 1
  return sdr


def corruptSparseVector(sdr, noiseLevel):
  """
  Add noise to sdr by turning off numNoiseBits active bits and turning on
  numNoiseBits in active bits
  @param sdr        (array) Numpy array of the  SDR
  @param noiseLevel (float) amount of noise to be applied on the vector.
  """
  numNoiseBits = int(noiseLevel * np.sum(sdr))
  if numNoiseBits <= 0:
    return sdr
  activeBits = np.where(sdr > 0)[0]
  inActiveBits = np.where(sdr == 0)[0]

  turnOffBits = np.random.permutation(activeBits)
  turnOnBits = np.random.permutation(inActiveBits)
  turnOffBits = turnOffBits[:numNoiseBits]
  turnOnBits = turnOnBits[:numNoiseBits]

  sdr[turnOffBits] = 0
  sdr[turnOnBits] = 1



def randomProjection(sdrs):
  """
  Project a set of N-dimensional SDRs to a 2D space with random matrix
  :param sdrs:
  :return:
  """
  numSDRs = len(sdrs)
  n = len(sdrs[0])

  randomMat = np.random.rand(n, 2)
  sdrsPos = np.zeros((numSDRs, 2))
  for i in range(numSDRs):
    sdrsPos[i, :] = (np.dot(sdrs[i].reshape((1, n)), randomMat))
  return sdrsPos



def plot_embedding(X, y, title=None):
  x_min, x_max = np.min(X, 0), np.max(X, 0)
  X = (X - x_min) / (x_max - x_min)

  plt.figure()
  ax = plt.subplot(111)
  colorList = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
  for i in range(X.shape[0]):
    plt.plot(X[i, 0], X[i, 1],
             colorList[y[i]]+'o')

  plt.xticks([]), plt.yticks([])
  if title is not None:
    plt.title(title)



if __name__ == "__main__":
  numSDRclasses = 5
  numSDRsPerClass = 20
  noiseLevel = 0.5

  # SDR parameters
  n = 1024
  w = 20

  sdrs = np.zeros((numSDRclasses*numSDRsPerClass, n))
  classLabel = np.zeros((numSDRclasses*numSDRsPerClass, ), dtype='int')
  for i in range(numSDRclasses):
    templateSDR = generateSDR(n, w)
    for j in range(numSDRsPerClass):
      noisySDR = copy.copy(templateSDR)
      corruptSparseVector(noisySDR, noiseLevel)
      sdrs[i*numSDRsPerClass+j, :] = noisySDR
      classLabel[i * numSDRsPerClass + j] = i

  numSDRs = sdrs.shape[0]
  # calculate pairwise distance
  distanceMat = np.zeros((numSDRs, numSDRs), dtype=np.float64)
  for i in range(numSDRs):
    for j in range(numSDRs):
      distanceMat[i, j] = 1 - percentOverlap(sdrs[i, ], sdrs[j, ])

  seed = np.random.RandomState(seed=3)

  # plot distance matrix
  plt.figure()
  plt.imshow(distanceMat)

  # visualize SDR clusters with MDS
  mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                     dissimilarity="precomputed", n_jobs=1)
  pos = mds.fit(distanceMat).embedding_

  nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
                      dissimilarity="precomputed", random_state=seed, n_jobs=1,
                      n_init=1)
  Xmds = nmds.fit_transform(distanceMat, init=pos)

  plot_embedding(Xmds, classLabel,
                 title='MDS, noise level: {}'.format(noiseLevel))
  plt.savefig('MDS_clusterN_{}_noiseLevel_{}.png'.format(numSDRclasses,
                                                         noiseLevel))

  # visualize SDR clusters with tSNE
  print("Computing t-SNE embedding")
  tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
  Xtsne = tsne.fit_transform(distanceMat)

  plot_embedding(Xtsne, classLabel,
                 title='tSNE, noise level: {}'.format(noiseLevel))
  plt.savefig('tSNE_clusterN_{}_noiseLevel_{}.png'.format(numSDRclasses,
                                                         noiseLevel))

  # visualize SDR clusters with random projection
  rp = random_projection.GaussianRandomProjection(n_components=2,
                                                  random_state=42)
  X_projected = rp.fit_transform(sdrs)
  plot_embedding(X_projected, classLabel,
                 title='Random Projection, noise level: {}'.format(noiseLevel))
  plt.savefig('RandomProj_clusterN_{}_noiseLevel_{}.png'.format(numSDRclasses,
                                                                noiseLevel))
