package me.ikasyk;

import mpi.MPI;

import java.util.Arrays;

public class Burgers {
    final int root = 0;
    final double A = 1;
    final double B = 2;
    final double tau = 0.1;
    final double h = 0.1;


    final double tn = 1;
    final double xn = 1;
    // MPI properties
    int processCount;
    int processId;
    private double[][] grid;
    private double[][] syst;

    private int nForX = (int) (xn / h) + 1;
    private int nForT = (int) (tn / tau) + 1;

    {
        grid = new double[nForT][nForX];
        syst = new double[nForX - 2][nForX + 1];
    }

    Burgers(String[] args) {
        MPI.Init(args);

        processCount = MPI.COMM_WORLD.Size();
        processId = MPI.COMM_WORLD.Rank();

        if (processId == 0) {
            for (int i = 0; i < nForX; i++) {
                grid[0][i] = f(i * h, 0);
            }
        }

        int partLength = (nForX - 2) / processCount;

        int[] recvcount = new int[processCount];
        int[] displs = new int[processCount];

        double[][] boundSystem = new double[2 * (processCount - 1)][nForX + 1];

        for (int i = 0; i < processCount; i++) {
            if (i == processCount - 1) {
                recvcount[i] = nForX - 2 - (partLength * (processCount - 1));
            } else {
                recvcount[i] = partLength;
            }
            displs[i] = partLength * i;
        }

        double sigma = tau / (h * h);

        int j = 1;
        while (j < nForT - 1) {
            /* Create system of linear equations */
            if (processId == 0) {
                for (int i = 0; i < nForX - 2; i++) {
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
                }
            }

            /* Free from elements below the main diagonal */
            MPI.COMM_WORLD.Scatterv(syst, 0, recvcount, displs, MPI.OBJECT, syst, displs[processId], recvcount[processId], MPI.OBJECT, root);
            for (int i = 1; i < recvcount[processId]; i++) {
                double oldA = syst[displs[processId] + i][displs[processId] + i];
                for (int k = 0; k < nForX + 1; k++) {
                    syst[displs[processId] + i][k] -= syst[displs[processId] + i - 1][k] / syst[displs[processId] + i - 1][displs[processId] + i] * oldA;

                }
            }

            /* Free from elements above the main diagonal */
            for (int i = recvcount[processId] - 2; i >= 0; i--) {
                double oldB = syst[displs[processId] + i][displs[processId] + i + 2];
                for (int k = 0; k < nForX + 1; k++) {
                    syst[displs[processId] + i][k] -= syst[displs[processId] + i + 1][k] / syst[displs[processId] + i + 1][displs[processId] + i + 2] * oldB;
                }
            }

            MPI.COMM_WORLD.Gatherv(syst, displs[processId], recvcount[processId], MPI.OBJECT, syst, 0, recvcount, displs, MPI.OBJECT, root);


            if (processId == 0) {

                /* Make and solve bound system */
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

            }


            /* Solve another equations between the bounds */

            MPI.COMM_WORLD.Scatterv(syst, 0, recvcount, displs, MPI.OBJECT, syst, displs[processId], recvcount[processId], MPI.OBJECT, root);

            if (processId != processCount - 1) {
                int toEqId = processId == 0 ? 0 : 1;
                for (int i = recvcount[processId] - 2; i >= toEqId; i--) {
                    double k = syst[displs[processId] + i][displs[processId + 1] + 1];
                    syst[displs[processId] + i][displs[processId + 1] + 1] = 0;
                    syst[displs[processId] + i][nForX] -= syst[displs[processId + 1] - 1][nForX] * k;
                }
            }
            MPI.COMM_WORLD.Gatherv(syst, displs[processId], recvcount[processId], MPI.OBJECT, syst, 0, recvcount, displs, MPI.OBJECT, root);


            if (processId != 0) {
                int toEqId = processId == processCount - 1 ? recvcount[processId] : recvcount[processId] - 1;
                for (int i = 1; i < toEqId; i++) {
                    double k = syst[displs[processId] + i][displs[processId]];
                    syst[displs[processId] + i][displs[processId]] = 0;
                    syst[displs[processId] + i][nForX] -= syst[displs[processId]][nForX] * k;
                }
            }

            int fromDiff = processId == 0 ? 0 : 1, toDiff = processId == processCount - 1 ? 1 : 0;
            for (int i = fromDiff; i < recvcount[processId] + toDiff - 1; i++) {
                syst[displs[processId] + i][nForX] /= syst[displs[processId] + i][displs[processId] + i + 1];
                syst[displs[processId] + i][displs[processId] + i + 1] = 1;
            }
            MPI.COMM_WORLD.Gatherv(syst, displs[processId], recvcount[processId], MPI.OBJECT, syst, 0, recvcount, displs, MPI.OBJECT, root);

            if (processId == 0) {
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
            }


            j++;
        }

        if (processId == 0) {
            for (int i = 0; i < nForT-1; i++) {
                System.out.println(Arrays.toString(grid[i]));
            }
        }

        MPI.COMM_WORLD.Barrier();
        MPI.Finalize();
    }

    public static void main(String[] args) {
        try {
            Burgers n = new Burgers(args);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private double f(double x, double t) {
        return (A - x) / (B + t);
    }
}
