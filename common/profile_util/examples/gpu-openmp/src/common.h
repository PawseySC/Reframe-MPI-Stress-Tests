/*! \file common.h
 *  \brief definitions useful for all files. 
 */

#ifndef COMMON_H
#define COMMON_H

#include <profile_util.h>

extern std::chrono::system_clock::time_point logtime;
extern std::time_t log_time;
extern char wherebuff[1000];
extern std::string whenbuff;

#define Where() sprintf(wherebuff,"@%s L%d ", __func__, __LINE__);
#define When() logtime = std::chrono::system_clock::now(); log_time = std::chrono::system_clock::to_time_t(logtime);whenbuff=std::ctime(&log_time);whenbuff.erase(std::find(whenbuff.begin(), whenbuff.end(), '\n'), whenbuff.end());
#define LocalLogger() Where();std::cout<<wherebuff<<" : " 


#endif
