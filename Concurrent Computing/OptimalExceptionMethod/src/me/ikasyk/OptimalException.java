package me.ikasyk;

import mpi.MPI;

import java.util.Arrays;

public class OptimalException {
    /*final double[][] A = {
            {5, 2, 3, 3},
            {1, 6, 1, 5},
            {3, -4, -2, 8}
    };*/
    final double[][] A = {
            {3, -1, -1, 4, -24},
            {2, 1, -3, 4, -26},
            {1, -1, 4, 1, 9},
            {4, 4, 3, -3, 21}
    };

    double[] X;

    {
        X = new double[A.length];
    }

    // MPI properties
    int processCount;
    int processId;
    final int root = 0;

    OptimalException(String[] args) {
        MPI.Init(args);

        processCount = MPI.COMM_WORLD.Size();
        processId = MPI.COMM_WORLD.Rank();

        int i;
        int n = A.length;

        double[][] tempA = new double[n][n+1];

        for (i = 0; i < n; i++) {
            tempA[i] = Arrays.copyOf(A[i], n+1);
        }

        int partLength = n / processCount;

        int[] recvcount = new int[processCount];
        int[] displs = new int[processCount];

        double[][] buf = new double[partLength+1][n+1];
        double bufi[] = new double[n+1];

        for (i = 0; i < processCount; i++) {
            if (i == processCount - 1) {
                recvcount[i] = n - (partLength * (processCount - 1));
            } else {
                recvcount[i] = partLength;
            }
            displs[i] = partLength * i;
        }

        i = 0;

        while (i < n) {
            if (processId == 0) {
                double ai = tempA[i][i];
                for (int j = 0; j < n+1; j++) {
                    tempA[i][j] /= ai;
                    bufi = Arrays.copyOf(tempA[i], n+1);
                }
            }

            MPI.COMM_WORLD.Bcast(bufi, 0, n+1, MPI.DOUBLE, root);
            MPI.COMM_WORLD.Scatterv(tempA, 0, recvcount, displs, MPI.OBJECT, buf, 0, recvcount[processId], MPI.OBJECT, root);


            for (int k = 0; k < recvcount[processId]; k++) {
                if (i != displs[processId] + k) {
                    double ai = buf[k][i];
                    for (int j = 0; j < n + 1; j++) {
                        buf[k][j] -= bufi[j] * ai;
                    }
                }
            }

            MPI.COMM_WORLD.Gatherv(buf, 0, recvcount[processId], MPI.OBJECT, tempA, 0, recvcount, displs, MPI.OBJECT, root);

            i++;
        }

        if (processId == 0) {
            for (i = 0; i < n; i++) {
                System.out.println("x" + (i+1) + " = "+tempA[i][n]);
            }
        }

        MPI.COMM_WORLD.Barrier();
        MPI.Finalize();
    }

    public static void main(String[] args) {
        try {
            OptimalException n = new OptimalException(args);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
