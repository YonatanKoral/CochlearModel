//    firpm_d
//    Copyright (C) 2015-2016  S. Filip
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.

//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.

//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>

#include "pm.h"
#include "band.h"
#include "barycentric.h"
#include "cvector.h"
#include <set>
#include <fstream>
#include <sstream>


double PMOGainAtOmega(const PMOutput& pmo, double omega) {
	
	return PMOGainAtOmega(pmo.h,omega);
}

double PMOGainAtOmega(const std::vector<double>& h, double omega) {
	double real_part = 0;
	double img_part = 0;
	int n = 0;
	for (double coefficent : h) {
		real_part += coefficent*cos(n*omega*M_PI);
		img_part += coefficent*sin(n*omega*M_PI);
	}
	return sqrt(real_part*real_part + img_part*img_part);
}
void fixDCGain(PMOutput& pmo, double targetGain) {
	double dcGain = PMOGainAtOmega(pmo, 0);
	pmo.h = (targetGain / dcGain)*pmo.h;
}

void fixDCGain(std::vector<double>& h, double targetGain) {
	double dcGain = PMOGainAtOmega(h, 0);
	h = (targetGain / dcGain)*h;
}
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXq;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXq;

void generateVandermondeMatrix(MatrixXq& A, std::size_t degree, std::vector<double>& meshPoints,
        std::function<double(double)>& weightFunction)
{

    A.resize(degree + 1u, meshPoints.size());
    for(std::size_t i = 0u; i < meshPoints.size(); ++i)
    {
        double pointWeight = weightFunction(meshPoints[i]);
        A(0u, i) = 1;
        A(1u, i) = meshPoints[i];
        for(std::size_t j = 2u; j <= degree; ++j)
            A(j, i) = meshPoints[i] * A(j - 1u, i) * 2 - A(j - 2u, i);
        for(std::size_t j = 0u; j <= degree; ++j)
            A(j, i) *= pointWeight;
    }
}

// approximate Fekete points
void AFP(std::vector<double>& points, MatrixXq& A, std::vector<double>& meshPoints)
{
    VectorXq b = VectorXq::Ones(A.rows());
    b(0) = 2;
    VectorXq y = A.colPivHouseholderQr().solve(b);


    for(std::size_t i = 0u; i < static_cast<size_t>(y.rows()); ++i)
        if(y(i) != 0.0)
            points.push_back(meshPoints[i]);
    std::sort(points.begin(), points.end(),
            [](const double& lhs,
               const double& rhs) {
                return lhs < rhs;
            });

}

void bandCount(std::vector<Band>& chebyBands, std::vector<double>& x)
{
    for(auto& it : chebyBands)
        it.extremas = 0u;
    std::size_t bandIt = 0u;
    for(std::size_t i = 0u; i < x.size(); ++i)
    {
        while(bandIt < chebyBands.size() && chebyBands[bandIt].stop < x[i])
            bandIt++;
        ++chebyBands[bandIt].extremas;
    }
}



void generateWAM(std::vector<double>& wam, std::vector<Band>& chebyBands, std::size_t degree)
{
    std::vector<double> chebyNodes(degree + 2u);
    generateEquidistantNodes(chebyNodes, degree + 1u);
    applyCos(chebyNodes, chebyNodes);
    for(std::size_t i = 0u; i < chebyBands.size(); ++i)
    {
        if(chebyBands[i].start != chebyBands[i].stop)
        {
            std::vector<double> bufferNodes(degree + 2u);
            changeOfVariable(bufferNodes, chebyNodes,
                    chebyBands[i].start, chebyBands[i].stop);
            for(auto& it : bufferNodes)
                wam.push_back(it);
        }
        else
            wam.push_back(chebyBands[i].start);
    }
}


void initUniformExtremas(std::vector<double>& omega,
        std::vector<Band>& B)
{
    double avgDistance = 0;

    std::vector<double> bandwidths(B.size());
    std::vector<std::size_t> nonPointBands;
    for(std::size_t i = 0; i < B.size(); ++i) {
        bandwidths[i] = B[i].stop - B[i].start;
        if(bandwidths[i] > 0.0)
        {
            nonPointBands.push_back(i);
            avgDistance += bandwidths[i];
        }
        B[i].extremas = 1u;
    }
    if(nonPointBands.empty())
    {
        std::cerr << "All intervals are points!\n";
        exit(EXIT_FAILURE);

    }
    // TODO: error check
    avgDistance /= (omega.size() - B.size());

    B[nonPointBands[nonPointBands.size() - 1u]].extremas = omega.size() - (B.size() - nonPointBands.size());
    double buffer;
    buffer = bandwidths[nonPointBands[0]] / avgDistance;
    buffer += 0.5;

        if (nonPointBands.size() > 1) {
            B[nonPointBands[0]].extremas = lrint(buffer) + 1;
            B[nonPointBands[nonPointBands.size() - 1u]].extremas -= B[nonPointBands[0]].extremas;
        }

        for(std::size_t i{1u}; i < nonPointBands.size() - 1; ++i) {
            buffer = bandwidths[nonPointBands[i]] / avgDistance;
            buffer += 0.5;
            B[nonPointBands[i]].extremas = lrint(buffer) + 1;
            B[nonPointBands[nonPointBands.size() - 1u]].extremas -= B[nonPointBands[i]].extremas;
        }


        std::size_t startIndex = 0ul;
        for(std::size_t i{0ul}; i < B.size(); ++i) {
            if(B[i].extremas > 1u)
                buffer = bandwidths[i] / (B[i].extremas - 1);
            omega[startIndex] = B[i].start;
            omega[startIndex + B[i].extremas - 1] = B[i].stop;
            for(std::size_t j{1ul}; j < B[i].extremas - 1; ++j)
                omega[startIndex + j] = omega[startIndex + j - 1] + buffer;
            startIndex += B[i].extremas;
        }
}

void referenceScaling(std::vector<double>& newX, std::vector<Band>& newChebyBands,
        std::vector<Band>& newFreqBands, std::size_t newXSize,
        std::vector<double>& x, std::vector<Band>& chebyBands,
        std::vector<Band>& freqBands)
{
        std::vector<std::size_t> newDistribution(chebyBands.size());
        for(std::size_t i{0u}; i < chebyBands.size(); ++i)
            newDistribution[i] = 0u;
        std::size_t multipointBands = 0u;
        std::size_t offset = 0u;
        int twoInt = 0;
        for(std::size_t i = 0u; i < chebyBands.size(); ++i)
        {
            newX.push_back(x[offset]);
            if(chebyBands[i].extremas > 2u)
            {
                ++multipointBands;
                for(std::size_t j = 1u; j < chebyBands[i].extremas - 2u; ++j)
                {
                    newX.push_back((x[offset + j] + x[offset + j + 1]) / 2);
                    newX.push_back(x[offset + j]);
                }
                newX.push_back(x[offset + chebyBands[i].extremas - 2u]);
                newX.push_back(x[offset + chebyBands[i].extremas - 1u]);
                twoInt += 2;
            }
            else if(chebyBands[i].extremas == 2u)
            {
                ++multipointBands;
                ++twoInt;
                newX.push_back(x[offset + 1u]);
                ++newDistribution[i];
            }
            offset += chebyBands[i].extremas;
        }
		int threeInt = static_cast<int>(newXSize - newX.size()) - twoInt;
        offset = 0u;
        for(std::size_t i = 0u; i < chebyBands.size(); ++i)
        {
                if(chebyBands[i].extremas > 1u)
                {
                    if(threeInt > 0)
                    {
                        newX.push_back(x[offset] + (x[offset + 1] - x[offset]) / 3);
                        double secondValue = x[offset] + (x[offset + 1] - x[offset]) / 3
                            + (x[offset + 1] - x[offset]) / 3;
                        newX.push_back(secondValue);
                        threeInt--;
                        twoInt--;
                    }
                    else if (twoInt > 0)
                    {
                        newX.push_back((x[offset] + x[offset + 1]) / 2);
                        twoInt--;
                    }
                }
            offset += chebyBands[i].extremas;
        }
        offset = 0;
        for(std::size_t i = 0u; i < chebyBands.size(); ++i)
        {
                if(chebyBands[i].extremas > 2u)
                {
                    if(threeInt > 0)
                    {
                        newX.push_back(x[offset + chebyBands[i].extremas - 2u] +
                                (x[offset + chebyBands[i].extremas - 1u] -
                                 x[offset + chebyBands[i].extremas - 2u]) / 3);
                        double secondValue = x[offset + chebyBands[i].extremas - 2u] +
                            (x[offset + chebyBands[i].extremas - 1u] -
                             x[offset + chebyBands[i].extremas - 2u]) / 3 +
                            (x[offset + chebyBands[i].extremas - 1u] -
                             x[offset + chebyBands[i].extremas - 2u]) / 3;
                        newX.push_back(secondValue);
                        threeInt--;
                        twoInt--;
                    }
                    else if (twoInt > 0)
                    {
                        newX.push_back((x[offset + chebyBands[i].extremas - 2u] +
                                    x[offset + chebyBands[i].extremas - 1u]) / 2);
                        twoInt--;
                    }
                }
            offset += chebyBands[i].extremas;
        }
        if(newXSize > newX.size())
        {
            std::cerr << "Failed to do reference scaling\n";
            exit(EXIT_FAILURE);
        }
        newX.resize(newXSize);
        std::sort(newX.begin(), newX.end());
        std::size_t total = 0u;
        for(std::size_t i = 0ul; i < newX.size(); ++i)
        {
                for(std::size_t j = 0u; j < chebyBands.size(); ++j)
                    if(newX[i] >= chebyBands[j].start && newX[i] <= chebyBands[j].stop)
                    {
                        newDistribution[j]++;
                        ++total;
                    }
        }
        if(total != newXSize)
        {
            std::cout << "Failed to find distribution!\n";
            exit(EXIT_FAILURE);
        }


        for (std::size_t i = 0u; i < chebyBands.size(); ++i)
        {
            newFreqBands[freqBands.size() - 1u - i].extremas = newDistribution[i];
            newChebyBands[i].extremas = newDistribution[i];
        }
}




void splitInterval(std::vector<Interval>& subIntervals,
        std::vector<Band>& chebyBands,
        std::vector<double> &x)
{
    std::size_t bandOffset = 0u;
    for(std::size_t i = 0u; i < chebyBands.size(); ++i)
    {
        if(bandOffset < x.size())
        {
            double middleValA, middleValB;
            if (x[bandOffset] > chebyBands[i].start
                && x[bandOffset] < chebyBands[i].stop)
            {
                middleValA = x[bandOffset];
                subIntervals.push_back(
                        std::make_pair(chebyBands[i].start, middleValA));
            } else {
                middleValA = chebyBands[i].start;
            }
            if(chebyBands[i].extremas > 1)
            {
                for(std::size_t j{bandOffset};
                    j < bandOffset + chebyBands[i].extremas - 1u; ++j)
                {
                    middleValB = x[j + 1];
                    subIntervals.push_back(std::make_pair(middleValA, middleValB));
                    middleValA = middleValB;
                }
                if(middleValA != chebyBands[i].stop)
                    subIntervals.push_back(
                        std::make_pair(middleValA, chebyBands[i].stop));
            }
            bandOffset += chebyBands[i].extremas;
        }
    }
}

void findEigenExtrema(double& convergenceOrder,
        double& delta, std::vector<double>& eigenExtrema,
        std::vector<double>& x, std::vector<Band>& chebyBands,
        int Nmax)
{
    // 1.   Split the initial [-1, 1] interval in subintervals
    //      in order that we can use a reasonable size matrix
    //      eigenvalue solver on the subintervals
    std::vector<Interval> subIntervals;
    double a = -1;
    double b = 1;

    splitInterval(subIntervals, chebyBands, x);

    //std::cout << "Number of subintervals: "
    //    << subIntervals.size() << std::endl;

    // 2.   Compute the barycentric variables (i.e. weights)
    //      needed for the current iteration

    std::vector<double> w(x.size());
    barycentricWeights(w, x);


    computeDelta(delta, w, x, chebyBands);
    //std::cout << "delta = " << delta << std::endl;

    std::vector<double> C(x.size());
    computeC(C, delta, x, chebyBands);

    // 3.   Use an eigenvalue solver on each subinterval to find the
    //      local extrema that are located inside the frequency bands
    std::vector<double> chebyNodes(Nmax + 1u);
    generateEquidistantNodes(chebyNodes, Nmax);
    applyCos(chebyNodes, chebyNodes);


    std::vector<std::pair<double, double>> potentialExtrema;
    std::vector<double> pEx;
    double extremaErrorValueLeft;
    double extremaErrorValueRight;
    double extremaErrorValue;
    computeError(extremaErrorValue, chebyBands[0].start,
            delta, x, C, w, chebyBands);
    potentialExtrema.push_back(std::make_pair(
            chebyBands[0].start, extremaErrorValue));


    for (std::size_t i = 0u; i < chebyBands.size() - 1; ++i)
    {
        computeError(extremaErrorValueLeft, chebyBands[i].stop,
                delta, x, C, w, chebyBands);
        computeError(extremaErrorValueRight, chebyBands[i + 1].start,
                delta, x, C, w, chebyBands);
        bool sgnLeft = std::signbit(extremaErrorValueLeft);
        bool sgnRight = std::signbit(extremaErrorValueRight);
        if (sgnLeft != sgnRight) {
            potentialExtrema.push_back(std::make_pair(
                    chebyBands[i].stop, extremaErrorValueLeft));
            potentialExtrema.push_back(std::make_pair(
                    chebyBands[i + 1].start, extremaErrorValueRight));
        } else {
            double abs1 = fabs(extremaErrorValueLeft);
            double abs2 = fabs(extremaErrorValueRight);
            if(abs1 > abs2)
                potentialExtrema.push_back(std::make_pair(
                        chebyBands[i].stop, extremaErrorValueLeft));
            else
                potentialExtrema.push_back(std::make_pair(
                        chebyBands[i + 1].start, extremaErrorValueRight));
        }
    }
    computeError(extremaErrorValue,
            chebyBands[chebyBands.size() - 1].stop,
            delta, x, C, w, chebyBands);
    potentialExtrema.push_back(std::make_pair(
            chebyBands[chebyBands.size() - 1].stop,
            extremaErrorValue));


    std::vector<std::vector<double>> pExs(subIntervals.size());

    #pragma omp parallel for
    for (std::size_t i = 0u; i < subIntervals.size(); ++i)
    {

        // find the Chebyshev nodes scaled to the current subinterval
        std::vector<double> siCN(Nmax + 1u);
        changeOfVariable(siCN, chebyNodes, subIntervals[i].first,
                subIntervals[i].second);

        // compute the Chebyshev interpolation function values on the
        // current subinterval
        std::vector<double> fx(Nmax + 1u);
        for (std::size_t j = 0u; j < fx.size(); ++j)
        {
            computeError(fx[j], siCN[j], delta, x, C, w,
                    chebyBands);

        }

        // compute the values of the CI coefficients and those of its
        // derivative
        std::vector<double> chebyCoeffs(Nmax + 1u);
        generateChebyshevCoefficients(chebyCoeffs, fx, Nmax);
        std::vector<double> derivCoeffs(Nmax);
        derivativeCoefficients2ndKind(derivCoeffs, chebyCoeffs);

        // solve the corresponding eigenvalue problem and determine the
        // local extrema situated in the current subinterval
        MatrixXq Cm(Nmax - 1u, Nmax - 1u);
        generateColleagueMatrix2ndKind(Cm, derivCoeffs);

        std::vector<double> eigenRoots;
        VectorXcq roots;
        determineEigenvalues(roots, Cm);
        getRealValues(eigenRoots, roots, a, b);
        changeOfVariable(eigenRoots, eigenRoots,
                subIntervals[i].first, subIntervals[i].second);
        for (std::size_t j = 0u; j < eigenRoots.size(); ++j)
            pExs[i].push_back(eigenRoots[j]);
        pExs[i].push_back(subIntervals[i].first);
        pExs[i].push_back(subIntervals[i].second);
    }

    for(std::size_t i = 0u; i < pExs.size(); ++i)
        for(std::size_t j = 0u; j < pExs[i].size(); ++j)
            pEx.push_back(pExs[i][j]);

    std::size_t startingOffset = potentialExtrema.size();
    potentialExtrema.resize(potentialExtrema.size() + pEx.size());
    #pragma omp parallel for
    for(std::size_t i = 0u; i < pEx.size(); ++i)
    {
        double valBuffer;
        computeError(valBuffer, pEx[i],
                delta, x, C, w, chebyBands);
        potentialExtrema[startingOffset + i] = std::make_pair(pEx[i], valBuffer);
    }

    // sort list of potential extrema in increasing order
    std::sort(potentialExtrema.begin(), potentialExtrema.end(),
            [](const std::pair<double, double>& lhs,
               const std::pair<double, double>& rhs) {
                return lhs.first < rhs.first;
            });

    eigenExtrema.clear();
    std::size_t extremaIt = 0u;
    std::vector<std::pair<double, double>> alternatingExtrema;
    double minError = INT_MAX;
    double maxError = INT_MIN;
    double absError;

    while (extremaIt < potentialExtrema.size())
    {
        std::pair<double, double> maxErrorPoint;
        maxErrorPoint = potentialExtrema[extremaIt];
        while(extremaIt < potentialExtrema.size() - 1 &&
            (std::signbit(maxErrorPoint.second) ==
             std::signbit(potentialExtrema[extremaIt + 1].second)))
        {
            ++extremaIt;
            if (fabs(maxErrorPoint.second) < fabs(potentialExtrema[extremaIt].second))
                maxErrorPoint = potentialExtrema[extremaIt];
        }
        alternatingExtrema.push_back(maxErrorPoint);
        ++extremaIt;
    }
    std::vector<std::pair<double, double>> bufferExtrema;
    //std::cout << "Alternating extrema: " << x.size() << " | "
    //    << alternatingExtrema.size() << std::endl;

    if(alternatingExtrema.size() < x.size())
    {
        std::cerr << "The exchange algorithm did not converge.\n";
        std::cerr << "TRIGGER: Not enough alternating extrema!\n"
            << "POSSIBLE CAUSE: Nmax too small\n";
        convergenceOrder = 2.0;
        return;
    }
    else if (alternatingExtrema.size() > x.size())
    {
        std::size_t remSuperfluous = alternatingExtrema.size() - x.size();
        if (remSuperfluous % 2 != 0)
        {
            if(remSuperfluous == 1u)
            {
                std::vector<double> x1, x2;
                x1.push_back(alternatingExtrema[0u].first);
                for(std::size_t i{1u}; i < alternatingExtrema.size() - 1; ++i)
                {
                    x1.push_back(alternatingExtrema[i].first);
                    x2.push_back(alternatingExtrema[i].first);
                }
                x2.push_back(alternatingExtrema[alternatingExtrema.size() - 1u].first);
                double delta1, delta2;
                computeDelta(delta1, x1, chebyBands);
                computeDelta(delta2, x2, chebyBands);
                delta1 = fabsl(delta1);
                delta2 = fabsl(delta2);
                std::size_t sIndex = 1u;
                if(delta1 > delta2)
                    sIndex = 0u;
                for(std::size_t i = sIndex; i < alternatingExtrema.size() + sIndex - 1u; ++i)
                    bufferExtrema.push_back(alternatingExtrema[i]);
                alternatingExtrema = bufferExtrema;
                bufferExtrema.clear();
            }
            else
            {
                double abs1 = fabs(alternatingExtrema[0].second);
                double abs2 = fabs(alternatingExtrema[alternatingExtrema.size() - 1].second);
                std::size_t sIndex = 0u;
                if (abs1 < abs2)
                    sIndex = 1u;
                for(std::size_t i = sIndex; i < alternatingExtrema.size() + sIndex - 1u; ++i)
                    bufferExtrema.push_back(alternatingExtrema[i]);
                alternatingExtrema = bufferExtrema;
                bufferExtrema.clear();
            }
        }


        while (alternatingExtrema.size() > x.size())
        {
            std::size_t toRemoveIndex = 0u;
            double minValToRemove = fminl(fabsl(alternatingExtrema[0].second),
                                              fabsl(alternatingExtrema[1].second));
            double removeBuffer;
            for (std::size_t i{1u}; i < alternatingExtrema.size() - 1; ++i)
            {
                removeBuffer = fminl(fabsl(alternatingExtrema[i].second),
                                   fabsl(alternatingExtrema[i + 1].second));
                if (removeBuffer < minValToRemove)
                {
                    minValToRemove = removeBuffer;
                    toRemoveIndex  = i;
                }
            }
            for (std::size_t i{0u}; i < toRemoveIndex; ++i)
                bufferExtrema.push_back(alternatingExtrema[i]);
            for (std::size_t i{toRemoveIndex + 2u}; i < alternatingExtrema.size(); ++i)
                bufferExtrema.push_back(alternatingExtrema[i]);
            alternatingExtrema = bufferExtrema;
            bufferExtrema.clear();
        }


    }
    if (alternatingExtrema.size() < x.size())
    {
        std::cerr << "Trouble!\n";
        exit(EXIT_FAILURE);
    }

    //std::cout << "After removal: " << alternatingExtrema.size() << std::endl;
    for (auto& it : alternatingExtrema)
    {
        eigenExtrema.push_back(it.first);
        absError = fabs(it.second);
        minError = fmin(minError, absError);
        maxError = fmax(maxError, absError);
    }

    //std::cout << "Min error = " << minError << std::endl;
    //std::cout << "Max error = " << maxError << std::endl;
    convergenceOrder = (maxError - minError) / maxError;
    //std::cout << "Convergence order = " << convergenceOrder << std::endl;
    // update the extrema count in each frequency band
    std::size_t bIndex = 0u;
    for(std::size_t i = 0; i < chebyBands.size(); ++i)
    {
        chebyBands[i].extremas = 0;
    }
    for(auto &it : eigenExtrema)
    {
        if(chebyBands[bIndex].start <= it && it <= chebyBands[bIndex].stop)
        {
            ++chebyBands[bIndex].extremas;
        }
        else
        {
            ++bIndex;
            ++chebyBands[bIndex].extremas;
        }
    }
}


// TODO: remember that this routine assumes that the information
// pertaining to the reference x and the frequency bands (i.e. the
// number of reference values inside each band) is given at the
// beginning of the execution
PMOutput exchange(std::vector<double>& x,
        std::vector<Band>& chebyBands, double eps,
        int Nmax)
{
    PMOutput output;

    std::size_t degree = x.size() - 2u;
    std::sort(x.begin(), x.end(),
            [](const double& lhs,
               const double& rhs) {
                return lhs < rhs;
            });
    std::vector<double> startX{x};
    std::cout.precision(20);

    output.Q = 1;
    output.iter = 0u;
    //double lastDelta = 1.0;
    do {
        ++output.iter;
        //std::cout << "*********ITERATION " << output.iter << " **********\n";
        findEigenExtrema(output.Q, output.delta,
                output.x, startX, chebyBands, Nmax);
        startX = output.x;
        if(output.Q > 1.0)
            break;
        //if(output.delta < lastDelta)
        //    break;
        //std::cout << "*********ITERATION " << output.iter << " **********\n";
    } while (output.Q > eps && output.iter <= 100u);

    if(isnan(output.delta) || isnan(output.Q))
        std::cerr << "The exchange algorithm did not converge.\n"
            << "TRIGGER: numerical instability\n"
            << "POSSIBLE CAUSES: poor starting reference and/or "
            << "a too small value for Nmax.\n";

    if(output.iter >= 101u)
        std::cerr << "The exchange algorithm did not converge.\n"
            << "TRIGGER: exceeded iteration threshold of 100\n"
            << "POSSIBLE CAUSES: poor starting reference and/or "
            << "a too small value for Nmax.\n";


    output.h.resize(degree + 1u);
    std::vector<double> finalC(output.x.size());
    std::vector<double> finalAlpha(output.x.size());
    barycentricWeights(finalAlpha, output.x);
    double finalDelta = output.delta;
    output.delta = fabsl(output.delta);
    //std::cout << "MINIMAX delta = " << output.delta << std::endl;
    computeC(finalC, finalDelta, output.x, chebyBands);
    std::vector<double> finalChebyNodes(degree + 1);
    generateEquidistantNodes(finalChebyNodes, degree);
    applyCos(finalChebyNodes, finalChebyNodes);
    std::vector<double> fv(degree + 1);

    for (std::size_t i{0u}; i < fv.size(); ++i)
        computeApprox(fv[i], finalChebyNodes[i], output.x,
                finalC, finalAlpha);

    generateChebyshevCoefficients(output.h, fv, degree);

    return output;
}


//---------------------------------------------------------------------------

// Goertzel is essentially a single frequency DFT, but without phase information.
// Its simplicity allows it to run about 3 times faster than a single frequency DFT.
// It is typically used to find a tone embedded in a signal. A DTMF tone for example.
// 256 pts in 6 us
double Goertzel(std::vector<double>const& Samples, double Omega)
{
	int j;
	double Reg0, Reg1, Reg2;        // 3 shift registers
	double CosVal, Mag;
	Reg1 = Reg2 = 0.0;
	int N = static_cast<int>(Samples.size());
	CosVal = 2.0 * cos(M_PI * Omega);
	for (j = 0; j<N; j++)
	{
		Reg0 = Samples[j] + CosVal * Reg1 - Reg2;
		Reg2 = Reg1;  // Shift the values.
		Reg1 = Reg0;
	}
	Mag = Reg2 * Reg2 + Reg1 * Reg1 - CosVal * Reg1 * Reg2;

	if (Mag > 0.0)Mag = sqrt(Mag);
	else Mag = 1.0E-12;

	return(Mag);
}

//---------------------------------------------------------------------------

// This function is used to correct the corner frequency values on FIR filters.
// We normally specify the 3 dB frequencies when specifing a filter. The Parks McClellan routine
// uses OmegaC and BW to set the 0 dB band edges, so its final OmegaC and BW values are not close
// to -3 dB. The Rectangular Windowed filters are better for high tap counts, but for low tap counts,
// their 3 dB frequencies are also well off the mark.

// To use this function, first calculate a set of FIR coefficients, then pass them here, along with
// OmegaC and BW. This calculates a corrected OmegaC for low and high pass filters. It calcultes a
// corrected BW for band pass and notch filters. Use these corrected values to recalculate the FIR filter.

// The Goertzel algorithm is used to calculate the filter's magnitude response at the single
// frequency defined in the loop. We start in the pass band and work out to the -20dB freq.
// target corner gain ensure correct corner gain to adjust omega
// modifie variation will need to be updated to seld detect if its bpf,lpf, hpf or notch
void FIRFreqError(std::vector<double>& h, int PassType, double *OmegaC, double targetCornerGain)
{
	int j, J3dB;
	double Omega, CorrectedOmega, Mag;
	double targetMagnitude = pow(10, -1 * targetCornerGain / 20);
	//std::cout << "target cordner gain DB : " << targetCornerGain << ",target magnitude: " << targetMagnitude << "\n";
	// In these loops, we break at -20 dB to ensure that large ripple is ignored.
	if (PassType == firLPF)
	{
		J3dB = 10;
		for (j = 0; j<NUM_FREQ_ERR_PTS; j++)
		{
			Omega = (double)j / dNUM_FREQ_ERR_PTS;
			//std::cout << "j: " << j << ",Omega:" << Omega << "\n";
			Mag = Goertzel(h, Omega);
			//std::cout << "Mag: " << Mag<< ",Omega:" << Omega << "\n";
			if (Mag > targetMagnitude)J3dB = j;   // J3dB will be the last j where the response was > -3 dB
			if (Mag < 0.1)break;        // Stop when the response is down to -20 dB.
		}
		Omega = (double)J3dB / dNUM_FREQ_ERR_PTS;
		//std::cout << "target cordner gain DB : " << targetCornerGain << ",target magnitude: " << targetMagnitude <<",Omega:" << Omega<< "\n";
	}

	else if (PassType == firHPF)
	{
		J3dB = NUM_FREQ_ERR_PTS - 10;
		for (j = NUM_FREQ_ERR_PTS - 1; j >= 0; j--)
		{
			Omega = (double)j / dNUM_FREQ_ERR_PTS;
			Mag = Goertzel(h, Omega);
			if (Mag > targetMagnitude)J3dB = j;  // J3dB will be the last j where the response was > -3 dB
			if (Mag < 0.1)break;       // Stop when the response is down to -20 dB.
		}
		Omega = (double)J3dB / dNUM_FREQ_ERR_PTS;
	}


	//std::cout << "target cordner gain DB : " << targetCornerGain << ",target magnitude: " << targetMagnitude << ",Omega:" << Omega << "\n";

	// This calculates the corrected OmegaC and BW and error checks the values.
	if (PassType == firLPF || PassType == firHPF)
	{
		CorrectedOmega = *OmegaC * 2.0 - Omega;  // This is usually OK.

		//std::cout << "target cordner gain DB : " << targetCornerGain << ",target magnitude: " << targetMagnitude << ",Omega:" << CorrectedOmega << "\n";
		if (CorrectedOmega < 0.001)CorrectedOmega = 0.001;
		if (CorrectedOmega > 0.99)CorrectedOmega = 0.99;
		//std::cout << "target cordner gain DB : " << targetCornerGain << ",target magnitude: " << targetMagnitude << ",Omega:" << CorrectedOmega << "\n";
		*OmegaC = CorrectedOmega;
	}
	//std::cout << "finally target cordner gain DB : " << targetCornerGain << ",target magnitude: " << targetMagnitude << ",Omega:" << CorrectedOmega << "\n";
}


void FIRFreqError(PMOutput& pmo, int PassType, double *OmegaC, double targetCornerGain) {
	FIRFreqError(pmo.h, PassType, OmegaC, targetCornerGain);
}


std::vector<Band> getFreqBandsForTypeIFPM(std::size_t n,
	std::vector<double>const& f,
	std::vector<double>const& a,
	std::vector<double>const& w) {
	std::vector<Band> freqBands(w.size());
	for (std::size_t i{ 0u }; i < freqBands.size(); ++i)
	{
		freqBands[i].start = M_PI * f[2u * i];
		if (i < freqBands.size() - 1u)
			freqBands[i].stop = M_PI * f[2u * i + 1u];
		else
		{
			if (f[2u * i + 1u] == 1.0)
			{
				if (f[2u * i] < 0.9999)
					freqBands[i].stop = M_PI * 0.9999;
				else
					freqBands[i].stop = M_PI * ((f[2u * i] + 1) / 2);
			}
			else
				freqBands[i].stop = M_PI * f[2u * i + 1u];
		}
		freqBands[i].space = BandSpace::FREQ;
		freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
		{
			if (a[2u * i] != a[2u * i + 1u]) {
				if (bSpace == BandSpace::CHEBY)
					x = acos(x);
				return (((x - freqBands[i].start) * a[2u * i + 1u] -
					(x - freqBands[i].stop) * a[2u * i]) /
					(freqBands[i].stop - freqBands[i].start)) / cos(x / 2);
			}
			if (bSpace == BandSpace::FREQ)
				return a[2u * i] / cos(x / 2);
			else
				return a[2u * i] / sqrt((x + 1) / 2);
		};
		freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
		{
			if (bSpace == BandSpace::FREQ)
				return cos(x / 2) * w[i];
			else
				return sqrt((x + 1) / 2) * w[i];
		};
	}
	return freqBands;
}

TFIRPassTypes typeofFIR(std::vector<Band> freqs) {
	if (freqs.size() > 3) return firNOT_FIR;
	else if (freqs.size() == 3) {
		if (freqs[0].amplitude(freqs[0].space, 0) == 1) return firNOTCH;
		else return firBPF;
	}
	else if (freqs.size() == 2) {
		if (freqs[0].amplitude(freqs[0].space, 0) == 1) return firLPF;
		else return firHPF;
	}
	return firNOT_FIR;
}


// type I&II filters
PMOutput firpm(std::size_t n,
        std::vector<double>const& f,
        std::vector<double>const& a,
        std::vector<double>const& w,
        double eps,
        int Nmax)
{
    std::vector<double> h;
    if( n % 2 != 0)
    {
        if((f[f.size() - 1u] == 1) && (a[a.size() - 1u] != 0))
        {
            std::cout << "Warning: Gain at Nyquist frequency different from 0.\n"
                << "Increasing the number of taps by 1 and passing to a "
                << " type I filter\n" << std::endl;
            ++n;
        } else {
            std::size_t degree = n / 2u;
            // TODO: error checking code
            std::vector<Band> freqBands(w.size());
            std::vector<Band> chebyBands;
            for(std::size_t i{0u}; i < freqBands.size(); ++i)
            {
                freqBands[i].start = M_PI * f[2u * i];
                if(i < freqBands.size() - 1u)
                    freqBands[i].stop  = M_PI * f[2u * i + 1u];
                else
                {
                    if(f[2u * i + 1u] == 1.0)
                    {
                        if(f[2u * i] < 0.9999)
                            freqBands[i].stop = M_PI * 0.9999;
                        else
                            freqBands[i].stop = M_PI * ((f[2u * i] + 1) / 2);
                    }
                    else
                        freqBands[i].stop  = M_PI * f[2u * i + 1u];
                }
                freqBands[i].space = BandSpace::FREQ;
                freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
                {
                    if (a[2u * i] != a[2u * i + 1u]) {
                        if(bSpace == BandSpace::CHEBY)
                            x = acos(x);
                        return (((x - freqBands[i].start) * a[2u * i + 1u] -
                                (x - freqBands[i].stop) * a[2u * i]) /
                                (freqBands[i].stop - freqBands[i].start)) / cos(x / 2);
                    }
                    if(bSpace == BandSpace::FREQ)
                        return a[2u * i] / cos(x / 2);
                    else
                        return a[2u * i] / sqrt((x + 1) / 2);
                };
                freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
                {
                    if (bSpace == BandSpace::FREQ)
                        return cos(x / 2) * w[i];
                    else
                        return sqrt((x + 1) / 2) * w[i];
                };
            }
            std::vector<double> omega(degree + 2u);
            std::vector<double> x(degree + 2u);
            initUniformExtremas(omega, freqBands);
            applyCos(x, omega);
            bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);


            PMOutput output = exchange(x, chebyBands, eps, Nmax);

            h.resize(n + 1u);
            h[0] = h[n] = output.h[degree] / 4;
            h[degree] = h[degree + 1] = (output.h[0] * 2 + output.h[1]) / 4;
            for(std::size_t i{2u}; i < degree + 1; ++i)
                h[degree + 1 - i] = h[degree + i] = (output.h[i - 1] + output.h[i]) / 4u;
            output.h = h;
            return output;
        }
    }


    std::size_t degree = n / 2u;
    // TODO: error checking code
    std::vector<Band> freqBands(w.size());
    std::vector<Band> chebyBands;
    for(std::size_t i{0u}; i < freqBands.size(); ++i)
    {
        freqBands[i].start = M_PI * f[2u * i];
        freqBands[i].stop  = M_PI * f[2u * i + 1u];
        freqBands[i].space = BandSpace::FREQ;
        freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
        {
            if (a[2u * i] != a[2u * i + 1u]) {
                if(bSpace == BandSpace::CHEBY)
                    x = acosl(x);
                return ((x - freqBands[i].start) * a[2u * i + 1u] -
                        (x - freqBands[i].stop) * a[2u * i]) /
                        (freqBands[i].stop - freqBands[i].start);
            }
            return a[2u * i];
        };
        freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
        {
            return w[i];
        };
    }

    std::vector<double> omega(degree + 2u);
    std::vector<double> x(degree + 2u);
    initUniformExtremas(omega, freqBands);
    applyCos(x, omega);
    bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

 //   double finalDelta;
    std::vector<double> coeffs;
    std::vector<double> finalExtrema;
//    double convergenceOrder;

    PMOutput output = exchange(x, chebyBands, eps, Nmax);

    h.resize(n + 1u);
    h[degree] = output.h[0];
    for(std::size_t i{0u}; i < degree; ++i)
        h[i] = h[n - i] = output.h[degree - i] / 2u;
    output.h = h;
    return output;

}


PMOutput firpmRS(std::size_t n,
        std::vector<double>const& f,
        std::vector<double>const& a,
        std::vector<double>const& w,
        double eps,
        std::size_t depth,
        int Nmax,
        RootSolver root)
{
    if (depth == 0u) return firpm(n, f, a, w, eps, Nmax);
    std::vector<double> h;
    if( n % 2 != 0)
    {
        if((f[f.size() - 1u] == 1) && (a[a.size() - 1u] != 0))
        {
            std::cout << "Warning: Gain at Nyquist frequency different from 0.\n"
                << "Increasing the number of taps by 1 and passing to a "
                << " type I filter\n" << std::endl;
            ++n;
        } else {
            std::size_t degree = n / 2u;
            // TODO: error checking code
            std::vector<Band> freqBands(w.size());
            std::vector<Band> chebyBands;
            for(std::size_t i{0u}; i < freqBands.size(); ++i)
            {
                freqBands[i].start = M_PI * f[2u * i];
                if(i < freqBands.size() - 1u)
                    freqBands[i].stop  = M_PI * f[2u * i + 1u];
                else
                {
                    if(f[2u * i + 1u] == 1.0)
                    {
                        if(f[2u * i] < 0.9999)
                            freqBands[i].stop = M_PI * 0.9999;
                        else
                            freqBands[i].stop = M_PI * ((f[2u * i] + 1) / 2);
                    }
                    else
                        freqBands[i].stop  = M_PI * f[2u * i + 1u];
                }
                freqBands[i].space = BandSpace::FREQ;
                freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
                {
                    if (a[2u * i] != a[2u * i + 1u]) {
                        if(bSpace == BandSpace::CHEBY)
                            x = acos(x);
                        return (((x - freqBands[i].start) * a[2u * i + 1u] -
                                (x - freqBands[i].stop) * a[2u * i]) /
                                (freqBands[i].stop - freqBands[i].start)) / cos(x / 2);
                    }
                    if(bSpace == BandSpace::FREQ)
                        return a[2u * i] / cos(x / 2);
                    else
                        return a[2u * i] / sqrt((x + 1) / 2);
                };
                freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
                {
                    if (bSpace == BandSpace::FREQ)
                        return cos(x / 2) * w[i];
                    else
                        return sqrt((x + 1) / 2) * w[i];
                };
            }

            std::vector<std::size_t> scaledDegrees(depth + 1u);
            scaledDegrees[depth] = degree;
			for (int i = static_cast<int>(depth)-1u; i >= 0; --i)
            {
                scaledDegrees[i] = scaledDegrees[i + 1] / 2;
            }

            std::vector<double> omega(scaledDegrees[0] + 2u);
            std::vector<double> x(scaledDegrees[0] + 2u);
            PMOutput output;
            if(root == RootSolver::UNIFORM) {
                initUniformExtremas(omega, freqBands);
                applyCos(x, omega);
                bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);
                output = exchange(x, chebyBands, eps, Nmax);

            } else {
                std::function<double(double)> weightFunction = [=](double x) -> double
                {
                    for(std::size_t i = 0u; i < chebyBands.size(); ++i)
                        if(chebyBands[i].start <= x && x <= chebyBands[i].stop)
							return chebyBands[i].weight(BandSpace::CHEBY, x);
					return 1; // for completion and avoid c4715 error
                };
                std::vector<double> wam;
                generateWAM(wam, chebyBands, degree);
                MatrixXq A;
                generateVandermondeMatrix(A, degree + 1u, wam, weightFunction);
                std::vector<double> afpX;
                AFP(afpX, A, wam);
                bandCount(chebyBands, afpX);

                output = exchange(afpX, chebyBands, eps, Nmax);
            }



            for(std::size_t i = 1u; i <= depth; ++i)
            {
                x.clear();
                referenceScaling(x, chebyBands, freqBands, scaledDegrees[i] + 2u,
                        output.x, chebyBands, freqBands);
                output = exchange(x, chebyBands, eps, Nmax);
            }

            h.resize(n + 1u);
            h[0] = h[n] = output.h[degree] / 4;
            h[degree] = h[degree + 1] = (output.h[0] * 2 + output.h[1]) / 4;
            for(std::size_t i{2u}; i < degree + 1; ++i)
                h[degree + 1 - i] = h[degree + i] = (output.h[i - 1] + output.h[i]) / 4u;
            output.h = h;
            return output;
        }
    }


    std::size_t degree = n / 2u;
    // TODO: error checking code
    std::vector<Band> freqBands(w.size());
    std::vector<Band> chebyBands;
    for(std::size_t i{0u}; i < freqBands.size(); ++i)
    {
        freqBands[i].start = M_PI * f[2u * i];
        freqBands[i].stop  = M_PI * f[2u * i + 1u];
        freqBands[i].space = BandSpace::FREQ;
        freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
        {
            if (a[2u * i] != a[2u * i + 1u]) {
                if(bSpace == BandSpace::CHEBY)
                    x = acosl(x);
                return ((x - freqBands[i].start) * a[2u * i + 1u] -
                        (x - freqBands[i].stop) * a[2u * i]) /
                        (freqBands[i].stop - freqBands[i].start);
            }
            return a[2u * i];
        };
        freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
        {
            return w[i];
        };
    }

    std::vector<std::size_t> scaledDegrees(depth + 1u);
    scaledDegrees[depth] = degree;
	for (int i = static_cast<int>(depth) - 1u; i >= 0; --i)
    {
        scaledDegrees[i] = scaledDegrees[i + 1] / 2;
    }

    std::vector<double> omega(scaledDegrees[0] + 2u);
    std::vector<double> x(scaledDegrees[0] + 2u);
    PMOutput output;
    if(root == RootSolver::UNIFORM) {
        initUniformExtremas(omega, freqBands);
        applyCos(x, omega);
        bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);
        output = exchange(x, chebyBands, eps, Nmax);

    } else {
        std::function<double(double)> weightFunction = [=](double x) -> double
        {
            for(std::size_t i = 0u; i < chebyBands.size(); ++i)
                if(chebyBands[i].start <= x && x <= chebyBands[i].stop)
					return chebyBands[i].weight(BandSpace::CHEBY, x);
			return 1; // for completion and avoid c4715 error
        };
        std::vector<double> wam;
        generateWAM(wam, chebyBands, degree);
        MatrixXq A;
        generateVandermondeMatrix(A, degree + 1u, wam, weightFunction);
        std::vector<double> afpX;
        AFP(afpX, A, wam);
        bandCount(chebyBands, afpX);

        output = exchange(afpX, chebyBands, eps, Nmax);
    }


    for(std::size_t i = 1u; i <= depth; ++i)
    {
        x.clear();
        referenceScaling(x, chebyBands, freqBands, scaledDegrees[i] + 2u,
                output.x, chebyBands, freqBands);
        output = exchange(x, chebyBands, eps, Nmax);
    }


    h.resize(n + 1u);
    h[degree] = output.h[0];
    for(std::size_t i{0u}; i < degree; ++i)
        h[i] = h[n - i] = output.h[degree - i] / 2u;
    output.h = h;
    return output;

}

// type III & IV filters
PMOutput firpm(std::size_t n,
        std::vector<double>const& f,
        std::vector<double>const& a,
        std::vector<double>const& w,
        ftype type,
        double eps,
        int Nmax)
{
    PMOutput output;
    std::vector<double> h;
    switch(type) {
        case ftype::FIR_DIFFERENTIATOR :
            {
                std::size_t degree = n / 2u;
                // TODO: error checking code
                 std::vector<double> fn = f;

                std::vector<Band> freqBands(w.size());
                std::vector<Band> chebyBands;
                double scaleFactor = a[1] / (f[1] * M_PI);
                if(n % 2 == 0) // Type III
                {
                    if(f[0u] == 0.0l)
                    {
                        if(fn[1u] < 0.00001l)
                            fn[0u] = fn[1] / 2;
                        else
                            fn[0u] = 0.00001l;
                    }
                    if(f[f.size() - 1u] == 1.0l)
                    {
                        if(f[f.size() - 2u] > 0.9999l)
                            fn[f.size() - 1u] = (1.0 + f[f.size() - 2u]) / 2;
                        else
                            fn[f.size() - 1u] = 0.9999l;
                    }
                    --degree;
                    freqBands[0].start = M_PI * fn[0u];
                    freqBands[0].stop  = M_PI * fn[1u];
                    freqBands[0].space = BandSpace::FREQ;
                    freqBands[0].weight = [w](BandSpace bSpace, double x) -> double
                    {
                        if(bSpace == BandSpace::FREQ)
                        {
                            return (sin(x) / x) * w[0u];
                        }
                        else
                        {
                            return (sqrt(1.0l - x * x) / acos(x)) * w[0u];
                        }
                    };
                    freqBands[0].amplitude = [scaleFactor](BandSpace bSpace, double x) -> double
                    {
                        if(bSpace == BandSpace::FREQ)
                        {
                            return (x / sin(x)) * scaleFactor;
                        }
                        else
                        {
                            return (acos(x) / sqrt(1.0l - x * x)) * scaleFactor;
                        }
                    };
                    for(std::size_t i{1u}; i < freqBands.size(); ++i)
                    {
                        freqBands[i].start = M_PI * fn[2u * i];
                        freqBands[i].stop  = M_PI * fn[2u * i + 1u];
                        freqBands[i].space = BandSpace::FREQ;
                        freqBands[i].weight = [w, i](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::FREQ)
                            {
                                return sin(x) * w[i];
                            }
                            else
                            {
                                return sqrt(1.0l - x * x) * w[i];
                            }

                        };
                        freqBands[i].amplitude = [freqBands, a, i](BandSpace bSpace, double x) -> double
                        {
                            if (a[2u * i] != a[2u * i + 1u]) {
                                if(bSpace == BandSpace::CHEBY)
                                    x = acos(x);
                                return ((x - freqBands[i].start) * a[2u * i + 1u] -
                                        (x - freqBands[i].stop) * a[2u * i]) /
                                        (freqBands[i].stop - freqBands[i].start);
                            }
                            return a[2u * i];

                        };

                    }

                } else {       // Type IV
                    if(f[0u] == 0.0l)
                    {
                        if(fn[1u] < 0.00001l)
                            fn[0u] = fn[1] / 2;
                        else
                            fn[0u] = 0.00001l;
                    }

                    freqBands[0].start = M_PI * fn[0u];
                    freqBands[0].stop  = M_PI * fn[1u];
                    freqBands[0].space = BandSpace::FREQ;
                    freqBands[0].weight = [w](BandSpace bSpace, double x) -> double
                    {
                        if(bSpace == BandSpace::FREQ)
                        {
                            return (sin(x / 2) / x) * w[0u];
                        }
                        else
                        {
                            return (sin(acos(x) / 2) / acos(x)) * w[0u];
                        }
                    };
                    freqBands[0].amplitude = [scaleFactor](BandSpace bSpace, double x) -> double
                    {
                        if(bSpace == BandSpace::FREQ)
                        {
                            return (x / sin(x / 2)) * scaleFactor;
                        }
                        else
                        {
                            return (acos(x) / sin(acos(x) / 2)) * scaleFactor;
                        }
                    };
                    for(std::size_t i{1u}; i < freqBands.size(); ++i)
                    {
                        freqBands[i].start = M_PI * fn[2u * i];
                        freqBands[i].stop  = M_PI * fn[2u * i + 1u];
                        freqBands[i].space = BandSpace::FREQ;
                        freqBands[i].weight = [w,i](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::FREQ)
                            {
                                return sin(x / 2) * w[i];
                            }
                            else
                            {
                                return (sin(acos(x) / 2)) * w[i];
                            }

                        };
                        freqBands[i].amplitude = [freqBands, a, i](BandSpace bSpace, double x) -> double
                        {
                            if (a[2u * i] != a[2u * i + 1u]) {
                                if(bSpace == BandSpace::CHEBY)
                                    x = acos(x);
                                return ((x - freqBands[i].start) * a[2u * i + 1u] -
                                        (x - freqBands[i].stop) * a[2u * i]) /
                                        (freqBands[i].stop - freqBands[i].start);
                            }
                            return a[2u * i];

                        };

                    }

                }

                std::vector<double> omega(degree + 2u);
                std::vector<double> x(degree + 2u);
                initUniformExtremas(omega, freqBands);
                applyCos(x, omega);
                bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

                output = exchange(x, chebyBands, eps, Nmax);

                h.resize(n + 1u);
                if(n % 2 == 0)
                {
                    h[degree + 1u] = 0;
                    h[degree] = (output.h[0u] * 2.0l - output.h[2]) / 4u;
                    h[degree + 2u] = -h[degree];
                    h[1u] = output.h[degree - 1u] / 4;
                    h[2u * degree + 1u] = -h[1u];
                    h[0u] =  output.h[degree] / 4;
                    h[2u * (degree + 1u)] = -h[0u];
                    for(std::size_t i{2u}; i < degree; ++i)
                    {
                        h[degree + 1u - i] = (output.h[i - 1u] - output.h[i + 1u]) / 4;
                        h[degree + 1u + i] = -h[degree + 1u - i];
                    }
                } else {
                    ++degree;
                    h[degree - 1u] = (output.h[0u] * 2.0l - output.h[1u]) / 4;
                    h[degree] = -h[degree - 1u];
                    h[0u] = output.h[degree - 1u] / 4;
                    h[2u * degree - 1u] = -h[0u];
                    for(std::size_t i{2u}; i < degree; ++i)
                    {
                        h[degree - i] = (output.h[i - 1u] - output.h[i]) / 4;
                        h[degree + i - 1u] = -h[degree - i];
                    }
                }


            }
            break;
        default : // FIR_HILBERT
            {
                std::size_t degree = n / 2u;
                std::vector<double> fn = f;
                // TODO: error checking code
                std::vector<Band> freqBands(w.size());
                std::vector<Band> chebyBands;
                if(n % 2 == 0) // Type III
                {
                    --degree;
                    if(f[0u] == 0.0l)
                    {
                        if(fn[1u] < 0.00001l)
                            fn[0u] = fn[1] / 2;
                        else
                            fn[0u] = 0.00001l;
                    }
                    if(f[f.size() - 1u] == 1.0l)
                    {
                        if(f[f.size() - 2u] > 0.9999l)
                            fn[f.size() - 1u] = (1.0 + f[f.size() - 2u]) / 2;
                        else
                            fn[f.size() - 1u] = 0.9999l;
                    }

                    for(std::size_t i{0u}; i < freqBands.size(); ++i)
                    {
                        freqBands[i].start = M_PI * fn[2u * i];
                        freqBands[i].stop  = M_PI * fn[2u * i + 1u];
                        freqBands[i].space = BandSpace::FREQ;
                        freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::CHEBY)
                                x = acosl(x);

                            if (a[2u * i] != a[2u * i + 1u]) {
                                return (((x - freqBands[i].start) * a[2u * i + 1u] -
                                        (x - freqBands[i].stop) * a[2u * i]) /
                                        (freqBands[i].stop - freqBands[i].start)) / sinl(x);
                            }
                            return a[2u * i] / sin(x);
                        };
                        freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::FREQ)
                                return w[i] * sin(x);
                            else
                                return w[i] * (sqrt(1.0l - x * x));
                        };
                    }
                } else {       // Type IV
                    if(f[0u] == 0.0l)
                    {
                        if(f[1u] < 0.00001l)
                            fn[0u] = fn[1u] / 2;
                        else
                            fn[0u] = 0.00001l;
                    }
                    for(std::size_t i{0u}; i < freqBands.size(); ++i)
                    {
                        freqBands[i].start = M_PI * fn[2u * i];
                        freqBands[i].stop  = M_PI * fn[2u * i + 1u];
                        freqBands[i].space = BandSpace::FREQ;
                        freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::CHEBY)
                                x = acos(x);

                            if (a[2u * i] != a[2u * i + 1u]) {
                                return (((x - freqBands[i].start) * a[2u * i + 1u] -
                                        (x - freqBands[i].stop) * a[2u * i]) /
                                        (freqBands[i].stop - freqBands[i].start)) / sinl(x / 2);
                            }
                            return a[2u * i] / sin(x / 2);
                        };
                        freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::FREQ)
                                return w[i] * sin(x / 2);
                            else
                            {
                                x = acos(x);
                                return w[i] * (sin(x / 2));
                            }
                        };
                    }
                }
                std::vector<double> omega(degree + 2u);
                std::vector<double> x(degree + 2u);
                initUniformExtremas(omega, freqBands);
                applyCos(x, omega);
                bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

                output = exchange(x, chebyBands, eps, Nmax);

                h.resize(n + 1u);
                if(n % 2 == 0)
                {
                    h[degree + 1u] = 0;
                    h[degree] = (output.h[0u] * 2.0l - output.h[2]) / 4u;
                    h[degree + 2u] = -h[degree];
                    h[1u] = output.h[degree - 1u] / 4;
                    h[2u * degree + 1u] = -h[1u];
                    h[0u] =  output.h[degree] / 4;
                    h[2u * (degree + 1u)] = -h[0u];
                    for(std::size_t i{2u}; i < degree; ++i)
                    {
                        h[degree + 1u - i] = (output.h[i - 1u] - output.h[i + 1u]) / 4;
                        h[degree + 1u + i] = -h[degree + 1u - i];
                    }
                } else {
                    ++degree;
                    h[degree - 1u] = (output.h[0u] * 2.0l - output.h[1u]) / 4;
                    h[degree] = -h[degree - 1u];
                    h[0u] = output.h[degree - 1u] / 4;
                    h[2u * degree - 1u] = -h[0u];
                    for(std::size_t i{2u}; i < degree; ++i)
                    {
                        h[degree - i] = (output.h[i - 1u] - output.h[i]) / 4;
                        h[degree + i - 1u] = -h[degree - i];
                    }
                }

            }
            break;
    }
    output.h = h;
    return output;
}


PMOutput firpmRS(std::size_t n,
        std::vector<double>const& f,
        std::vector<double>const& a,
        std::vector<double>const& w,
        ftype type,
        double eps,
        std::size_t depth,
        int Nmax,
        RootSolver root)
{
    if (depth == 0u) return firpm(n, f, a, w, type, eps, Nmax);
    PMOutput output;
    std::vector<double> h;
    switch(type) {
        case ftype::FIR_DIFFERENTIATOR :
            {
                std::size_t degree = n / 2u;
                // TODO: error checking code
                std::vector<double> fn = f;

                std::vector<Band> freqBands(w.size());
                std::vector<Band> chebyBands;
                double scaleFactor = a[1] / (f[1] * M_PI);
                if(n % 2 == 0) // Type III
                {
                    if(f[0u] == 0.0l)
                    {
                        if(fn[1u] < 0.00001l)
                            fn[0u] = fn[1] / 2;
                        else
                            fn[0u] = 0.00001l;
                    }
                    if(f[f.size() - 1u] == 1.0l)
                    {
                        if(f[f.size() - 2u] > 0.9999l)
                            fn[f.size() - 1u] = (1.0 + f[f.size() - 2u]) / 2;
                        else
                            fn[f.size() - 1u] = 0.9999l;
                    }
                    --degree;
                    freqBands[0].start = M_PI * fn[0u];
                    freqBands[0].stop  = M_PI * fn[1u];
                    freqBands[0].space = BandSpace::FREQ;
                    freqBands[0].weight = [w](BandSpace bSpace, double x) -> double
                    {
                        if(bSpace == BandSpace::FREQ)
                        {
                            return (sin(x) / x) * w[0u];
                        }
                        else
                        {
                            return (sqrt(1.0l - x * x) / acos(x)) * w[0u];
                        }
                    };
                    freqBands[0].amplitude = [scaleFactor](BandSpace bSpace, double x) -> double
                    {
                        if(bSpace == BandSpace::FREQ)
                        {
                            return (x / sin(x)) * scaleFactor;
                        }
                        else
                        {
                            return (acos(x) / sqrt(1.0l - x * x)) * scaleFactor;
                        }
                    };
                    for(std::size_t i{1u}; i < freqBands.size(); ++i)
                    {
                        freqBands[i].start = M_PI * fn[2u * i];
                        freqBands[i].stop  = M_PI * fn[2u * i + 1u];
                        freqBands[i].space = BandSpace::FREQ;
                        freqBands[i].weight = [w, i](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::FREQ)
                            {
                                return sin(x) * w[i];
                            }
                            else
                            {
                                return sqrt(1.0l - x * x) * w[i];
                            }

                        };
                        freqBands[i].amplitude = [freqBands, a, i](BandSpace bSpace, double x) -> double
                        {
                            if (a[2u * i] != a[2u * i + 1u]) {
                                if(bSpace == BandSpace::CHEBY)
                                    x = acos(x);
                                return ((x - freqBands[i].start) * a[2u * i + 1u] -
                                        (x - freqBands[i].stop) * a[2u * i]) /
                                        (freqBands[i].stop - freqBands[i].start);
                            }
                            return a[2u * i];

                        };

                    }

                } else {       // Type IV
                    if(f[0u] == 0.0l)
                    {
                        if(fn[1u] < 0.00001l)
                            fn[0u] = fn[1] / 2;
                        else
                            fn[0u] = 0.00001l;
                    }

                    freqBands[0].start = M_PI * fn[0u];
                    freqBands[0].stop  = M_PI * fn[1u];
                    freqBands[0].space = BandSpace::FREQ;
                    freqBands[0].weight = [w](BandSpace bSpace, double x) -> double
                    {
                        if(bSpace == BandSpace::FREQ)
                        {
                            return (sin(x / 2) / x) * w[0u];
                        }
                        else
                        {
                            return (sin(acos(x) / 2) / acos(x)) * w[0u];
                        }
                    };
                    freqBands[0].amplitude = [scaleFactor](BandSpace bSpace, double x) -> double
                    {
                        if(bSpace == BandSpace::FREQ)
                        {
                            return (x / sin(x / 2)) * scaleFactor;
                        }
                        else
                        {
                            return (acos(x) / sin(acos(x) / 2)) * scaleFactor;
                        }
                    };
                    for(std::size_t i{1u}; i < freqBands.size(); ++i)
                    {
                        freqBands[i].start = M_PI * fn[2u * i];
                        freqBands[i].stop  = M_PI * fn[2u * i + 1u];
                        freqBands[i].space = BandSpace::FREQ;
                        freqBands[i].weight = [w,i](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::FREQ)
                            {
                                return sin(x / 2) * w[i];
                            }
                            else
                            {
                                return (sin(acos(x) / 2)) * w[i];
                            }

                        };
                        freqBands[i].amplitude = [freqBands, a, i](BandSpace bSpace, double x) -> double
                        {
                            if (a[2u * i] != a[2u * i + 1u]) {
                                if(bSpace == BandSpace::CHEBY)
                                    x = acos(x);
                                return ((x - freqBands[i].start) * a[2u * i + 1u] -
                                        (x - freqBands[i].stop) * a[2u * i]) /
                                        (freqBands[i].stop - freqBands[i].start);
                            }
                            return a[2u * i];

                        };

                    }

                }

            std::vector<std::size_t> scaledDegrees(depth + 1u);
            scaledDegrees[depth] = degree;
			for (int i = static_cast<int>(depth) - 1u; i >= 0; --i)
            {
                scaledDegrees[i] = scaledDegrees[i + 1] / 2;
            }

            std::vector<double> omega(scaledDegrees[0] + 2u);
            std::vector<double> x(scaledDegrees[0] + 2u);
            PMOutput output;
            if(root == RootSolver::UNIFORM) {
                initUniformExtremas(omega, freqBands);
                applyCos(x, omega);
                bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);
                output = exchange(x, chebyBands, eps, Nmax);

            } else {
                std::function<double(double)> weightFunction = [=](double x) -> double
                {
                    for(std::size_t i = 0u; i < chebyBands.size(); ++i)
                        if(chebyBands[i].start <= x && x <= chebyBands[i].stop)
							return chebyBands[i].weight(BandSpace::CHEBY, x);
					return 1; // for completion and avoid c4715 error
                };
                std::vector<double> wam;
                generateWAM(wam, chebyBands, degree);
                MatrixXq A;
                generateVandermondeMatrix(A, degree + 1u, wam, weightFunction);
                std::vector<double> afpX;
                AFP(afpX, A, wam);
                bandCount(chebyBands, afpX);

                output = exchange(afpX, chebyBands, eps, Nmax);
            }


            for(std::size_t i = 1u; i <= depth; ++i)
            {
                x.clear();
                referenceScaling(x, chebyBands, freqBands, scaledDegrees[i] + 2u,
                        output.x, chebyBands, freqBands);
                output = exchange(x, chebyBands, eps, Nmax);
            }

                h.resize(n + 1u);
                if(n % 2 == 0)
                {
                    h[degree + 1u] = 0;
                    h[degree] = (output.h[0u] * 2.0l - output.h[2]) / 4u;
                    h[degree + 2u] = -h[degree];
                    h[1u] = output.h[degree - 1u] / 4;
                    h[2u * degree + 1u] = -h[1u];
                    h[0u] =  output.h[degree] / 4;
                    h[2u * (degree + 1u)] = -h[0u];
                    for(std::size_t i{2u}; i < degree; ++i)
                    {
                        h[degree + 1u - i] = (output.h[i - 1u] - output.h[i + 1u]) / 4;
                        h[degree + 1u + i] = -h[degree + 1u - i];
                    }
                } else {
                    ++degree;
                    h[degree - 1u] = (output.h[0u] * 2.0l - output.h[1u]) / 4;
                    h[degree] = -h[degree - 1u];
                    h[0u] = output.h[degree - 1u] / 4;
                    h[2u * degree - 1u] = -h[0u];
                    for(std::size_t i{2u}; i < degree; ++i)
                    {
                        h[degree - i] = (output.h[i - 1u] - output.h[i]) / 4;
                        h[degree + i - 1u] = -h[degree - i];
                    }
                }


            }
            break;
        default : // FIR_HILBERT
            {
                std::size_t degree = n / 2u;
                std::vector<double> fn = f;
                // TODO: error checking code
                std::vector<Band> freqBands(w.size());
                std::vector<Band> chebyBands;
                if(n % 2 == 0) // Type III
                {
                    --degree;
                    if(f[0u] == 0.0l)
                    {
                        if(fn[1u] < 0.00001l)
                            fn[0u] = fn[1] / 2;
                        else
                            fn[0u] = 0.00001l;
                    }
                    if(f[f.size() - 1u] == 1.0l)
                    {
                        if(f[f.size() - 2u] > 0.9999l)
                            fn[f.size() - 1u] = (1.0 + f[f.size() - 2u]) / 2;
                        else
                            fn[f.size() - 1u] = 0.9999l;
                    }

                    for(std::size_t i{0u}; i < freqBands.size(); ++i)
                    {
                        freqBands[i].start = M_PI * fn[2u * i];
                        freqBands[i].stop  = M_PI * fn[2u * i + 1u];
                        freqBands[i].space = BandSpace::FREQ;
                        freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::CHEBY)
                                x = acosl(x);

                            if (a[2u * i] != a[2u * i + 1u]) {
                                return (((x - freqBands[i].start) * a[2u * i + 1u] -
                                        (x - freqBands[i].stop) * a[2u * i]) /
                                        (freqBands[i].stop - freqBands[i].start)) / sinl(x);
                            }
                            return a[2u * i] / sin(x);
                        };
                        freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::FREQ)
                                return w[i] * sin(x);
                            else
                                return w[i] * (sqrt(1.0l - x * x));
                        };
                    }
                } else {       // Type IV
                    if(f[0u] == 0.0l)
                    {
                        if(f[1u] < 0.00001l)
                            fn[0u] = fn[1u] / 2;
                        else
                            fn[0u] = 0.00001l;
                    }
                    for(std::size_t i{0u}; i < freqBands.size(); ++i)
                    {
                        freqBands[i].start = M_PI * fn[2u * i];
                        freqBands[i].stop  = M_PI * fn[2u * i + 1u];
                        freqBands[i].space = BandSpace::FREQ;
                        freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::CHEBY)
                                x = acos(x);

                            if (a[2u * i] != a[2u * i + 1u]) {
                                return (((x - freqBands[i].start) * a[2u * i + 1u] -
                                        (x - freqBands[i].stop) * a[2u * i]) /
                                        (freqBands[i].stop - freqBands[i].start)) / sinl(x / 2);
                            }
                            return a[2u * i] / sin(x / 2);
                        };
                        freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::FREQ)
                                return w[i] * sin(x / 2);
                            else
                            {
                                x = acos(x);
                                return w[i] * (sin(x / 2));
                            }
                        };
                    }
                }



                std::vector<std::size_t> scaledDegrees(depth + 1u);
                scaledDegrees[depth] = degree;
				for (int i = static_cast<int>(depth) - 1u; i >= 0; --i)
                {
                    scaledDegrees[i] = scaledDegrees[i + 1] / 2;
                }

                std::vector<double> omega(scaledDegrees[0] + 2u);
                std::vector<double> x(scaledDegrees[0] + 2u);
                PMOutput output;
                if(root == RootSolver::UNIFORM) {
                    initUniformExtremas(omega, freqBands);
                    applyCos(x, omega);
                    bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);
                    output = exchange(x, chebyBands, eps, Nmax);

                } else {
                    std::function<double(double)> weightFunction = [=](double x) -> double
                    {
                        for(std::size_t i = 0u; i < chebyBands.size(); ++i)
                            if(chebyBands[i].start <= x && x <= chebyBands[i].stop)
                                return chebyBands[i].weight(BandSpace::CHEBY, x);
						return 1; // for completion and avoid c4715 error
                    };
                    std::vector<double> wam;
                    generateWAM(wam, chebyBands, degree);
                    MatrixXq A;
                    generateVandermondeMatrix(A, degree + 1u, wam, weightFunction);
                    std::vector<double> afpX;
                    AFP(afpX, A, wam);
                    bandCount(chebyBands, afpX);

                    output = exchange(afpX, chebyBands, eps, Nmax);
                }



                for(std::size_t i = 1u; i <= depth; ++i)
                {
                    x.clear();
                    referenceScaling(x, chebyBands, freqBands, scaledDegrees[i] + 2u,
                            output.x, chebyBands, freqBands);
                    output = exchange(x, chebyBands, eps, Nmax);
                }

                h.resize(n + 1u);
                if(n % 2 == 0)
                {
                    h[degree + 1u] = 0;
                    h[degree] = (output.h[0u] * 2.0l - output.h[2]) / 4u;
                    h[degree + 2u] = -h[degree];
                    h[1u] = output.h[degree - 1u] / 4;
                    h[2u * degree + 1u] = -h[1u];
                    h[0u] =  output.h[degree] / 4;
                    h[2u * (degree + 1u)] = -h[0u];
                    for(std::size_t i{2u}; i < degree; ++i)
                    {
                        h[degree + 1u - i] = (output.h[i - 1u] - output.h[i + 1u]) / 4;
                        h[degree + 1u + i] = -h[degree + 1u - i];
                    }
                } else {
                    ++degree;
                    h[degree - 1u] = (output.h[0u] * 2.0l - output.h[1u]) / 4;
                    h[degree] = -h[degree - 1u];
                    h[0u] = output.h[degree - 1u] / 4;
                    h[2u * degree - 1u] = -h[0u];
                    for(std::size_t i{2u}; i < degree; ++i)
                    {
                        h[degree - i] = (output.h[i - 1u] - output.h[i]) / 4;
                        h[degree + i - 1u] = -h[degree - i];
                    }
                }

            }
            break;
    }
    output.h = h;
    return output;
}



// type I&II filters
PMOutput firpmAFP(std::size_t n,
        std::vector<double>const& f,
        std::vector<double>const& a,
        std::vector<double>const& w,
        double eps,
        int Nmax)
{
    std::vector<double> h;
    if( n % 2 != 0)
    {
        if((f[f.size() - 1u] == 1) && (a[a.size() - 1u] != 0))
        {
            std::cout << "Warning: Gain at Nyquist frequency different from 0.\n"
                << "Increasing the number of taps by 1 and passing to a "
                << " type I filter\n" << std::endl;
            ++n;
        } else {
            std::size_t degree = n / 2u;
            // TODO: error checking code
            std::vector<Band> freqBands(w.size());
            std::vector<Band> chebyBands;
            for(std::size_t i{0u}; i < freqBands.size(); ++i)
            {
                freqBands[i].start = M_PI * f[2u * i];
                if(i < freqBands.size() - 1u)
                    freqBands[i].stop  = M_PI * f[2u * i + 1u];
                else
                {
                    if(f[2u * i + 1u] == 1.0)
                    {
                        if(f[2u * i] < 0.9999)
                            freqBands[i].stop = M_PI * 0.9999;
                        else
                            freqBands[i].stop = M_PI * ((f[2u * i] + 1) / 2);
                    }
                    else
                        freqBands[i].stop  = M_PI * f[2u * i + 1u];
                }
                freqBands[i].space = BandSpace::FREQ;
                freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
                {
                    if (a[2u * i] != a[2u * i + 1u]) {
                        if(bSpace == BandSpace::CHEBY)
                            x = acos(x);
                        return (((x - freqBands[i].start) * a[2u * i + 1u] -
                                (x - freqBands[i].stop) * a[2u * i]) /
                                (freqBands[i].stop - freqBands[i].start)) / cos(x / 2);
                    }
                    if(bSpace == BandSpace::FREQ)
                        return a[2u * i] / cos(x / 2);
                    else
                        return a[2u * i] / sqrt((x + 1) / 2);
                };
                freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
                {
                    if (bSpace == BandSpace::FREQ)
                        return cos(x / 2) * w[i];
                    else
                        return sqrt((x + 1) / 2) * w[i];
                };
            }
            bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);
            std::function<double(double)> weightFunction = [=](double x) -> double
            {
                for(std::size_t i = 0u; i < chebyBands.size(); ++i)
                    if(chebyBands[i].start <= x && x <= chebyBands[i].stop)
						return chebyBands[i].weight(BandSpace::CHEBY, x);
				return 1; // for completion and avoid c4715 error
            };
            std::vector<double> wam;
            generateWAM(wam, chebyBands, degree);
            MatrixXq A;
            generateVandermondeMatrix(A, degree + 1u, wam, weightFunction);
            std::vector<double> afpX;
            AFP(afpX, A, wam);
            bandCount(chebyBands, afpX);


            PMOutput output = exchange(afpX, chebyBands, eps, Nmax);

            h.resize(n + 1u);
            h[0] = h[n] = output.h[degree] / 4;
            h[degree] = h[degree + 1] = (output.h[0] * 2 + output.h[1]) / 4;
            for(std::size_t i{2u}; i < degree + 1; ++i)
                h[degree + 1 - i] = h[degree + i] = (output.h[i - 1] + output.h[i]) / 4u;
            output.h = h;
            return output;
        }
    }


    std::size_t degree = n / 2u;
    // TODO: error checking code
    std::vector<Band> freqBands(w.size());
    std::vector<Band> chebyBands;
    for(std::size_t i{0u}; i < freqBands.size(); ++i)
    {
        freqBands[i].start = M_PI * f[2u * i];
        freqBands[i].stop  = M_PI * f[2u * i + 1u];
        freqBands[i].space = BandSpace::FREQ;
        freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
        {
            if (a[2u * i] != a[2u * i + 1u]) {
                if(bSpace == BandSpace::CHEBY)
                    x = acosl(x);
                return ((x - freqBands[i].start) * a[2u * i + 1u] -
                        (x - freqBands[i].stop) * a[2u * i]) /
                        (freqBands[i].stop - freqBands[i].start);
            }
            return a[2u * i];
        };
        freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
        {
            return w[i];
        };
    }

    bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);
    std::function<double(double)> weightFunction = [=](double x) -> double
    {
        for(std::size_t i = 0u; i < chebyBands.size(); ++i)
            if(chebyBands[i].start <= x && x <= chebyBands[i].stop)
				return chebyBands[i].weight(BandSpace::CHEBY, x);
		return 1; // for completion and avoid c4715 error
    };
    std::vector<double> wam;
    generateWAM(wam, chebyBands, degree);
    MatrixXq A;
    generateVandermondeMatrix(A, degree + 1u, wam, weightFunction);
    std::vector<double> afpX;
    AFP(afpX, A, wam);
    bandCount(chebyBands, afpX);



  //  double finalDelta;
    std::vector<double> coeffs;
    std::vector<double> finalExtrema;
//    double convergenceOrder;

    PMOutput output = exchange(afpX, chebyBands, eps, Nmax);

    h.resize(n + 1u);
    h[degree] = output.h[0];
    for(std::size_t i{0u}; i < degree; ++i)
        h[i] = h[n - i] = output.h[degree - i] / 2u;
    output.h = h;
    return output;

}

// type III & IV filters
PMOutput firpmAFP(std::size_t n,
        std::vector<double>const& f,
        std::vector<double>const& a,
        std::vector<double>const& w,
        ftype type,
        double eps,
        int Nmax)
{
    PMOutput output;
    std::vector<double> h;
    switch(type) {
        case ftype::FIR_DIFFERENTIATOR :
            {
                std::size_t degree = n / 2u;
                // TODO: error checking code
                 std::vector<double> fn = f;

                std::vector<Band> freqBands(w.size());
                std::vector<Band> chebyBands;
                double scaleFactor = a[1] / (f[1] * M_PI);
                if(n % 2 == 0) // Type III
                {
                    if(f[0u] == 0.0l)
                    {
                        if(fn[1u] < 0.00001l)
                            fn[0u] = fn[1] / 2;
                        else
                            fn[0u] = 0.00001l;
                    }
                    if(f[f.size() - 1u] == 1.0l)
                    {
                        if(f[f.size() - 2u] > 0.9999l)
                            fn[f.size() - 1u] = (1.0 + f[f.size() - 2u]) / 2;
                        else
                            fn[f.size() - 1u] = 0.9999l;
                    }
                    --degree;
                    freqBands[0].start = M_PI * fn[0u];
                    freqBands[0].stop  = M_PI * fn[1u];
                    freqBands[0].space = BandSpace::FREQ;
                    freqBands[0].weight = [w](BandSpace bSpace, double x) -> double
                    {
                        if(bSpace == BandSpace::FREQ)
                        {
                            return (sin(x) / x) * w[0u];
                        }
                        else
                        {
                            return (sqrt(1.0l - x * x) / acos(x)) * w[0u];
                        }
                    };
                    freqBands[0].amplitude = [scaleFactor](BandSpace bSpace, double x) -> double
                    {
                        if(bSpace == BandSpace::FREQ)
                        {
                            return (x / sin(x)) * scaleFactor;
                        }
                        else
                        {
                            return (acos(x) / sqrt(1.0l - x * x)) * scaleFactor;
                        }
                    };
                    for(std::size_t i{1u}; i < freqBands.size(); ++i)
                    {
                        freqBands[i].start = M_PI * fn[2u * i];
                        freqBands[i].stop  = M_PI * fn[2u * i + 1u];
                        freqBands[i].space = BandSpace::FREQ;
                        freqBands[i].weight = [w, i](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::FREQ)
                            {
                                return sin(x) * w[i];
                            }
                            else
                            {
                                return sqrt(1.0l - x * x) * w[i];
                            }

                        };
                        freqBands[i].amplitude = [freqBands, a, i](BandSpace bSpace, double x) -> double
                        {
                            if (a[2u * i] != a[2u * i + 1u]) {
                                if(bSpace == BandSpace::CHEBY)
                                    x = acos(x);
                                return ((x - freqBands[i].start) * a[2u * i + 1u] -
                                        (x - freqBands[i].stop) * a[2u * i]) /
                                        (freqBands[i].stop - freqBands[i].start);
                            }
                            return a[2u * i];

                        };

                    }

                } else {       // Type IV
                    if(f[0u] == 0.0l)
                    {
                        if(fn[1u] < 0.00001l)
                            fn[0u] = fn[1] / 2;
                        else
                            fn[0u] = 0.00001l;
                    }

                    freqBands[0].start = M_PI * fn[0u];
                    freqBands[0].stop  = M_PI * fn[1u];
                    freqBands[0].space = BandSpace::FREQ;
                    freqBands[0].weight = [w](BandSpace bSpace, double x) -> double
                    {
                        if(bSpace == BandSpace::FREQ)
                        {
                            return (sin(x / 2) / x) * w[0u];
                        }
                        else
                        {
                            return (sin(acos(x) / 2) / acos(x)) * w[0u];
                        }
                    };
                    freqBands[0].amplitude = [scaleFactor](BandSpace bSpace, double x) -> double
                    {
                        if(bSpace == BandSpace::FREQ)
                        {
                            return (x / sin(x / 2)) * scaleFactor;
                        }
                        else
                        {
                            return (acos(x) / sin(acos(x) / 2)) * scaleFactor;
                        }
                    };
                    for(std::size_t i{1u}; i < freqBands.size(); ++i)
                    {
                        freqBands[i].start = M_PI * fn[2u * i];
                        freqBands[i].stop  = M_PI * fn[2u * i + 1u];
                        freqBands[i].space = BandSpace::FREQ;
                        freqBands[i].weight = [w,i](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::FREQ)
                            {
                                return sin(x / 2) * w[i];
                            }
                            else
                            {
                                return (sin(acos(x) / 2)) * w[i];
                            }

                        };
                        freqBands[i].amplitude = [freqBands, a, i](BandSpace bSpace, double x) -> double
                        {
                            if (a[2u * i] != a[2u * i + 1u]) {
                                if(bSpace == BandSpace::CHEBY)
                                    x = acos(x);
                                return ((x - freqBands[i].start) * a[2u * i + 1u] -
                                        (x - freqBands[i].stop) * a[2u * i]) /
                                        (freqBands[i].stop - freqBands[i].start);
                            }
                            return a[2u * i];

                        };

                    }

                }

                bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);
                std::function<double(double)> weightFunction = [=](double x) -> double
                {
                    for(std::size_t i = 0u; i < chebyBands.size(); ++i)
                        if(chebyBands[i].start <= x && x <= chebyBands[i].stop)
							return chebyBands[i].weight(BandSpace::CHEBY, x);
					return 1; // for completion and avoid c4715 error
                };
                std::vector<double> wam;
                generateWAM(wam, chebyBands, degree);
                MatrixXq A;
                generateVandermondeMatrix(A, degree + 1u, wam, weightFunction);
                std::vector<double> afpX;
                AFP(afpX, A, wam);
                bandCount(chebyBands, afpX);


                output = exchange(afpX, chebyBands, eps, Nmax);

                h.resize(n + 1u);
                if(n % 2 == 0)
                {
                    h[degree + 1u] = 0;
                    h[degree] = (output.h[0u] * 2.0l - output.h[2]) / 4u;
                    h[degree + 2u] = -h[degree];
                    h[1u] = output.h[degree - 1u] / 4;
                    h[2u * degree + 1u] = -h[1u];
                    h[0u] =  output.h[degree] / 4;
                    h[2u * (degree + 1u)] = -h[0u];
                    for(std::size_t i{2u}; i < degree; ++i)
                    {
                        h[degree + 1u - i] = (output.h[i - 1u] - output.h[i + 1u]) / 4;
                        h[degree + 1u + i] = -h[degree + 1u - i];
                    }
                } else {
                    ++degree;
                    h[degree - 1u] = (output.h[0u] * 2.0l - output.h[1u]) / 4;
                    h[degree] = -h[degree - 1u];
                    h[0u] = output.h[degree - 1u] / 4;
                    h[2u * degree - 1u] = -h[0u];
                    for(std::size_t i{2u}; i < degree; ++i)
                    {
                        h[degree - i] = (output.h[i - 1u] - output.h[i]) / 4;
                        h[degree + i - 1u] = -h[degree - i];
                    }
                }


            }
            break;
        default : // FIR_HILBERT
            {
                std::size_t degree = n / 2u;
                std::vector<double> fn = f;
                // TODO: error checking code
                std::vector<Band> freqBands(w.size());
                std::vector<Band> chebyBands;
                if(n % 2 == 0) // Type III
                {
                    --degree;
                    if(f[0u] == 0.0l)
                    {
                        if(fn[1u] < 0.00001l)
                            fn[0u] = fn[1] / 2;
                        else
                            fn[0u] = 0.00001l;
                    }
                    if(f[f.size() - 1u] == 1.0l)
                    {
                        if(f[f.size() - 2u] > 0.9999l)
                            fn[f.size() - 1u] = (1.0 + f[f.size() - 2u]) / 2;
                        else
                            fn[f.size() - 1u] = 0.9999l;
                    }

                    for(std::size_t i{0u}; i < freqBands.size(); ++i)
                    {
                        freqBands[i].start = M_PI * fn[2u * i];
                        freqBands[i].stop  = M_PI * fn[2u * i + 1u];
                        freqBands[i].space = BandSpace::FREQ;
                        freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::CHEBY)
                                x = acosl(x);

                            if (a[2u * i] != a[2u * i + 1u]) {
                                return (((x - freqBands[i].start) * a[2u * i + 1u] -
                                        (x - freqBands[i].stop) * a[2u * i]) /
                                        (freqBands[i].stop - freqBands[i].start)) / sinl(x);
                            }
                            return a[2u * i] / sin(x);
                        };
                        freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::FREQ)
                                return w[i] * sin(x);
                            else
                                return w[i] * (sqrt(1.0l - x * x));
                        };
                    }
                } else {       // Type IV
                    if(f[0u] == 0.0l)
                    {
                        if(f[1u] < 0.00001l)
                            fn[0u] = fn[1u] / 2;
                        else
                            fn[0u] = 0.00001l;
                    }
                    for(std::size_t i{0u}; i < freqBands.size(); ++i)
                    {
                        freqBands[i].start = M_PI * fn[2u * i];
                        freqBands[i].stop  = M_PI * fn[2u * i + 1u];
                        freqBands[i].space = BandSpace::FREQ;
                        freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::CHEBY)
                                x = acos(x);

                            if (a[2u * i] != a[2u * i + 1u]) {
                                return (((x - freqBands[i].start) * a[2u * i + 1u] -
                                        (x - freqBands[i].stop) * a[2u * i]) /
                                        (freqBands[i].stop - freqBands[i].start)) / sinl(x / 2);
                            }
                            return a[2u * i] / sin(x / 2);
                        };
                        freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::FREQ)
                                return w[i] * sin(x / 2);
                            else
                            {
                                x = acos(x);
                                return w[i] * (sin(x / 2));
                            }
                        };
                    }
                }
                bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);
                std::function<double(double)> weightFunction = [=](double x) -> double
                {
                    for(std::size_t i = 0u; i < chebyBands.size(); ++i)
                        if(chebyBands[i].start <= x && x <= chebyBands[i].stop)
							return chebyBands[i].weight(BandSpace::CHEBY, x);
					return 1; // for completion and avoid c4715 error
                };
                std::vector<double> wam;
                generateWAM(wam, chebyBands, degree);
                MatrixXq A;
                generateVandermondeMatrix(A, degree + 1u, wam, weightFunction);
                std::vector<double> afpX;
                AFP(afpX, A, wam);
                bandCount(chebyBands, afpX);


                output = exchange(afpX, chebyBands, eps, Nmax);

                h.resize(n + 1u);
                if(n % 2 == 0)
                {
                    h[degree + 1u] = 0;
                    h[degree] = (output.h[0u] * 2.0l - output.h[2]) / 4u;
                    h[degree + 2u] = -h[degree];
                    h[1u] = output.h[degree - 1u] / 4;
                    h[2u * degree + 1u] = -h[1u];
                    h[0u] =  output.h[degree] / 4;
                    h[2u * (degree + 1u)] = -h[0u];
                    for(std::size_t i{2u}; i < degree; ++i)
                    {
                        h[degree + 1u - i] = (output.h[i - 1u] - output.h[i + 1u]) / 4;
                        h[degree + 1u + i] = -h[degree + 1u - i];
                    }
                } else {
                    ++degree;
                    h[degree - 1u] = (output.h[0u] * 2.0l - output.h[1u]) / 4;
                    h[degree] = -h[degree - 1u];
                    h[0u] = output.h[degree - 1u] / 4;
                    h[2u * degree - 1u] = -h[0u];
                    for(std::size_t i{2u}; i < degree; ++i)
                    {
                        h[degree - i] = (output.h[i - 1u] - output.h[i]) / 4;
                        h[degree + i - 1u] = -h[degree - i];
                    }
                }

            }
            break;
    }
    output.h = h;
    return output;
}
