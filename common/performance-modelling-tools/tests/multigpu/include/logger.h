/*! \file logger.h
 *  \brief report some info 
 */

#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <iostream> 
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>


class Logger
{
    public:
    void ReportTimes(std::string , std::vector<double> &);
    std::string ReportGPUSetup();
};

#endif