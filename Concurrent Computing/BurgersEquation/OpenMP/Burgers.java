import java.util.Arrays;

public class Burgers {
    static final int root = 0;
    static final double A = 1;
    static final double B = 2;
    static final double tau = 0.05;
    static final double h = 0.05;

    static final int processCount = 8;

    static final double tn = 1;
    static final double xn = 1;

    static private double[][] grid;
    static private double[][] syst;

    static private int nForX = (int) (xn / h) + 1;
    static private int nForT = (int) (tn / tau) + 1;

    static {
        grid = new double[nForT][nForX];
        syst = new double[nForX - 2][nForX + 1];
    }

    public static void main(String[] args) {
        // omp parallel for
        for (int i = 0; i < nForX; i++) {
            grid[0][i] = f(i * h, 0);
        }
        int processCount = 1;
        int processId = 1;

        // omp parallel
        {
            processCount = OMP4J_NUM_THREADS;
            // processId = OMP4J_THREAD_NUM;
        }

        int partLength = (nForX - 2) / processCount;
        int[] recvcount = new int[processCount];
        int[] displs = new int[processCount];
        // omp parallel
        {
            for (int i = 0; i < OMP4J_NUM_THREADS; i++) {
                if (i == OMP4J_NUM_THREADS - 1) {
                    recvcount[i] = nForX - 2 - (partLength * (OMP4J_NUM_THREADS - 1));
                } else {
                    recvcount[i] = partLength;
                }
                displs[i] = partLength * i;
            }
        }

        double[][] boundSystem = new double[2 * (processCount - 1)][nForX + 1];
        // omp sections
        {
            // omp section
            {

                // partLength = (nForX - 2) / OMP4J_NUM_THREADS;
                //
                // boundSystem = new double[2 * (OMP4J_NUM_THREADS - 1)][nForX + 1];
            }
        }
        double sigma = tau / (h * h);
        //
        int j = 1;
        // double uij;
        while (j <= nForT - 1) {
            /* Create system of linear equations */


            for (int i = 0; i < nForX - 2; i++) {
//                    double uij = f(i * h, j * tau);
                double uij = grid[j - 1][i];
                syst[i][i] = -sigma;
                syst[i][i + 1] = 1 + 2 * sigma + sigma * h * uij;
                syst[i][i + 2] = -sigma * (1 + h * uij);
                if (i == 0) {
                    syst[0][0] = 0;
                    syst[0][nForX] = uij + sigma * f(0, j * tau);
                } else if (i == nForX - 3) {
                    syst[i][nForX - 1] = 0;
                    syst[i][nForX] = uij + sigma * (1 + h * uij) * f(xn, j * tau);
                } else {
                    syst[i][nForX] = uij;
                }
                //   System.out.println("Initial: " + Arrays.toString(syst[i]));

            }

            // omp parallel
            {
                // System.out.println("revcount = "+Arrays.toString(displs));
                /* Free from elements below the main diagonal */
                for (int i = 1; i < recvcount[OMP4J_THREAD_NUM]; i++) {
                    double oldA = syst[displs[OMP4J_THREAD_NUM] + i][displs[OMP4J_THREAD_NUM] + i];
                    for (int k = 0; k < nForX + 1; k++) {
                        syst[displs[OMP4J_THREAD_NUM] + i][k] -= syst[displs[OMP4J_THREAD_NUM] + i - 1][k] / syst[displs[OMP4J_THREAD_NUM] + i - 1][displs[OMP4J_THREAD_NUM] + i] * oldA;
                    }
                }
            }

            // omp parallel
            {
                /* Free from elements above the main diagonal */
                for (int i = recvcount[OMP4J_THREAD_NUM] - 2; i >= 0; i--) {
                    double oldB = syst[displs[OMP4J_THREAD_NUM] + i][displs[OMP4J_THREAD_NUM] + i + 2];
                    for (int k = 0; k < nForX + 1; k++) {
                        syst[displs[OMP4J_THREAD_NUM] + i][k] -= syst[displs[OMP4J_THREAD_NUM] + i + 1][k] / syst[displs[OMP4J_THREAD_NUM] + i + 1][displs[OMP4J_THREAD_NUM] + i + 2] * oldB;
                    }
                }
            }
                                    /* omp parallel threadNum(1)
                                    {
                                        for (int i = 0; i < nForX-2; i++) {
                                            System.out.println(Arrays.toString(syst[i]));
                                        }
                                    }*/


            for (int i = 0; i < processCount - 1; i++) {
                boundSystem[i * 2] = Arrays.copyOf(syst[displs[i + 1] - 1], nForX + 1);
                boundSystem[i * 2 + 1] = Arrays.copyOf(syst[displs[i + 1]], nForX + 1);
            }
            for (int i = 0; i < processCount - 1; i++) {
                for (int equ = 0; equ <= 1; equ++) {
                    double c1 = boundSystem[i * 2 + equ][displs[i + 1] + equ];
                    for (int k = 0; k < nForX + 1; k++) {
                        boundSystem[i * 2 + equ][k] /= c1;
                    }

                    for (int l = 0; l < processCount - 1; l++) {
                        for (int equSubstract = 0; equSubstract <= 1; equSubstract++) {
                            if (i * 2 + equ != l * 2 + equSubstract) {
                                double ai = boundSystem[l * 2 + equSubstract][displs[i + 1] + equ];
                                for (int k = 0; k < nForX + 1; k++) {
                                    boundSystem[l * 2 + equSubstract][k] -= boundSystem[i * 2 + equ][k] * ai;
                                }
                            }
                        }
                    }
                }
            }
            for (int i = 0; i < processCount - 1; i++) {
                for (int equ = 0; equ <= 1; equ++)
                    syst[displs[i + 1] - 1 + equ] = Arrays.copyOf(boundSystem[i * 2 + 1 - equ], nForX + 1);
            }


            // omp parallel
            {
                if (OMP4J_THREAD_NUM != OMP4J_NUM_THREADS - 1) {
                    int toEqId = OMP4J_THREAD_NUM == 0 ? 0 : 1;
                    for (int i = recvcount[OMP4J_THREAD_NUM] - 2; i >= toEqId; i--) {
                        double k = syst[displs[OMP4J_THREAD_NUM] + i][displs[OMP4J_THREAD_NUM + 1] + 1];
                        syst[displs[OMP4J_THREAD_NUM] + i][displs[OMP4J_THREAD_NUM + 1] + 1] = 0;
                        syst[displs[OMP4J_THREAD_NUM] + i][nForX] -= syst[displs[OMP4J_THREAD_NUM + 1] - 1][nForX] * k;
                    }
                }
            }

            // omp parallel

            {

                if (OMP4J_THREAD_NUM != 0) {
                    int toEqId = OMP4J_THREAD_NUM == processCount - 1 ? recvcount[OMP4J_THREAD_NUM] : recvcount[OMP4J_THREAD_NUM] - 1;
                    for (int i = 1; i < toEqId; i++) {
                        double k = syst[displs[OMP4J_THREAD_NUM] + i][displs[OMP4J_THREAD_NUM]];
                        syst[displs[OMP4J_THREAD_NUM] + i][displs[OMP4J_THREAD_NUM]] = 0;
                        syst[displs[OMP4J_THREAD_NUM] + i][nForX] -= syst[displs[OMP4J_THREAD_NUM]][nForX] * k;
                    }
                }
                int fromDiff = OMP4J_THREAD_NUM == 0 ? 0 : 1, toDiff = OMP4J_THREAD_NUM == OMP4J_NUM_THREADS - 1 ? 1 : 0;
                for (int i = fromDiff; i < recvcount[OMP4J_THREAD_NUM] + toDiff - 1; i++) {
                    syst[displs[OMP4J_THREAD_NUM] + i][nForX] /= syst[displs[OMP4J_THREAD_NUM] + i][displs[OMP4J_THREAD_NUM] + i + 1];
                    syst[displs[OMP4J_THREAD_NUM] + i][displs[OMP4J_THREAD_NUM] + i + 1] = 1;
                }
            }


            for (int i = 0; i < processCount - 1; i++) {
                double[] temp = syst[displs[i + 1]];
                syst[displs[i + 1]] = syst[displs[i + 1] - 1];
                syst[displs[i + 1] - 1] = temp;
            }

            grid[j][0] = f(0, j * tau);
            grid[j][nForX - 1] = f(xn, j * tau);
            for (int i = 0; i < nForX - 2; i++) {
                grid[j][i + 1] = syst[i][nForX];
            }


            j++;
        }

            for (int i = 0; i < nForT-1; i++) {
                System.out.println(Arrays.toString(grid[i]));
            }
        
    }

    static private double f(double x, double t) {
        return (A - x) / (B + t);
    }
}
