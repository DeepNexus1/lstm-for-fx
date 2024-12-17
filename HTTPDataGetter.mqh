//#property strict

#import "HTTPGetLibrary.dll"
   void RequestData(const string url); // initiate of the request for async data getting
   bool DataWasReceived(const string url); // returns true when the data was received

   bool RequestWasFailed(const string url); // returns true if for some reason the request was failed
   void GetErrorMessage(const string url, string &result); // returns the message with the error message

   int GetDataValuesCount(const string url); // returns the count of the JSON data values
   bool GetDataValue(const string url, const int index, string &result); // returns the JSON data value at specified index

   bool AnyExceptionWasCaught(); //returns true if there was caught any common exception
   void GetLastExceptionText(string &result); // returns and resets the text of the last caught common exception
#import

#define MAX_STRING_LENGTH (5000) // max allowed strings length returned from C# dll

class HTTPDataGetter
{
   private:
      const ENUM_TIMEFRAMES m_timeframe;
      const string m_url;

      bool m_isDataReadyToUse;
      bool m_waitingForDataReceiving;
      datetime m_lastDataRequestingTime;

   public:
      HTTPDataGetter(const ENUM_TIMEFRAMES timeframe, const string url)
      :
         m_timeframe(timeframe),
         m_url(url),
         m_isDataReadyToUse(false),
         m_waitingForDataReceiving(false),
         m_lastDataRequestingTime((datetime)0)
      {
      }

      // updates data when there incomes a new candle on specified timeframe
      void UpdateDataIfRequired()
      {
         const datetime lastCandleTimeOpen = iTime(Symbol(), m_timeframe, 0);

         const bool isItTimeToRequestData = (m_lastDataRequestingTime != lastCandleTimeOpen);

         if(isItTimeToRequestData)
         {
            m_isDataReadyToUse = false;
            m_waitingForDataReceiving = false;
         }

         if (!m_isDataReadyToUse)
         {
            NotifyAboutCommonExceptionIfItHappened();

            // if we're not waiting for the data receiving, then request the data
            if(!m_waitingForDataReceiving)
            {
               HTTPGetLibrary::RequestData(m_url);

               m_waitingForDataReceiving = true;

               m_lastDataRequestingTime = lastCandleTimeOpen;

               return;
            }
            else
            {
               // if the data receiving was failed, then print the error message, cancel the data receiving
               // and mark the data as not ready to use
               if(HTTPGetLibrary::RequestWasFailed(m_url))
               {
                  string errorMessage = "";
                  StringInit(errorMessage, MAX_STRING_LENGTH, 0);

                  HTTPGetLibrary::GetErrorMessage(m_url, errorMessage);

                  Print("<<< error: ", errorMessage);

                  m_isDataReadyToUse = false;
                  m_waitingForDataReceiving = false;
               }

               // if the data was received, then cancel the data receiving  and mark the data as ready to use
               else if(HTTPGetLibrary::DataWasReceived(m_url))
               {
                  int dataValuesCount = HTTPGetLibrary::GetDataValuesCount(m_url);

                  Print("<<< ", TimeLocal(), ": data received, JSON values count = ", dataValuesCount);
                  

                  m_isDataReadyToUse = true;
                  m_waitingForDataReceiving = false;
               }
            }
         }
      }

      // returns true if the data is received and ready to use
      bool IsDataReadyToUse() const
      {
         return m_isDataReadyToUse;
      }

      // returns true if the data item at specified index was got, the data item (as double value) places into &value argument
      bool TryGetDataValue(const int index, double &value)
      {
         string valueStr = "";
         StringInit(valueStr, MAX_STRING_LENGTH, 0);

         if(HTTPGetLibrary::GetDataValue(m_url, index, valueStr))
         {
            value = StringToDouble(valueStr);
            

            return true;
         }
         else
         {
            value = -1.0;

            return false;
         }
         
      }

   private:
      void NotifyAboutCommonExceptionIfItHappened()
      {
         if(HTTPGetLibrary::AnyExceptionWasCaught())
         {
            string exceptionText = "";
            StringInit(exceptionText, MAX_STRING_LENGTH, 0);

            HTTPGetLibrary::GetLastExceptionText(exceptionText);
            if(exceptionText != "")
            {
               Print("Exception was caught, ", exceptionText);
            }
         }
      }
};