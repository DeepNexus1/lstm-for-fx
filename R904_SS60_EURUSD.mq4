
//+------------------------------------------------------------------+
//|                                             R904_SS60_EURUSD.mq4 |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+


/*-------------------------------------------------------------------------------
 Trading Strategy Code
 Population member: 467
 Max bars back: 25

 Created:    7/15/2019 15:21:53

 Scripting language: MetaTrader 4 (MQL4)

 Symbol file:  .csv
 Build dates: 1/21/2016 to 1/18/2018

 Project file: .gpstrat
-------------------------------------------------------------------------------*/
#include <AdaptradeBuilderInclude.mqh>

#define STRATORDERID  19715242

// Strategy inputs
extern int Shift1 = 9;             // Indicator shift value (bars)
extern int Shift2 = 2;             // Indicator shift value (bars)
extern int Shift3 = 17;            // Indicator shift value (bars)

//original code: extern double MMStopPct = 6.343;   // Value of percentage money management stop
extern double MMStopPct = 1.971;   // Value of percentage money management stop -- testing manual override value

extern int NATR = 25;              // Indicator look-back length (bars)

// original code: extern double ATRFrTrail = 2.9671; // Multiple of price difference (e.g., ATR); exit
extern double ATRFrTrail = 4.6210;

//original code: extern double TrailPct = 43.0000;  // Trailing stop percentage
extern double TrailPct = 95.000;

extern double PSParam = 100000.00; // Position sizing parameter value
extern bool RoundPS = true;        // Round-to-nearest (true/false)
extern int RoundTo = 1;            // Round-to position size value
extern int MinSize = 1;            // Minimum allowable position size
extern int SizeLimit = 100000;     // Maximum allowable position size

// Global strategy variables
int MaxBarsBack = 25;
double PointValue = 1.000000;
int MaxSlippage = 3;
double SharesPerLot = 100000;
datetime LastBarTime = Time[0]; //original strategy code
// added this:
datetime LastActionTime;
/* +-------
// from this site: https://mql4tradingautomation.com/mql4-on-new-bar/
datetime NewBarTime = TimeCurrent(); //trying something different here for bar time

bool isNewBar() {
   if(NewBarTime==iTime(Symbol(),0,0)) return false;
   
   else{
      NewBarTime=iTime(Symbol(),0,0);
      return true;
   }
}
// +--------


// from https://www.mql5.com/en/forum/160180
int start()
{
   if( isNewBar() )
   {
      GetRequest(0);
   }
   return(0);
}
--------------------------------+
*/

// Variables for exit prices
double LStop = 0;
double NewLStop = 0;
bool LTrailOn = false;
double SStop = 0;
double NewSStop = 0;
bool STrailOn = false;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- create timer 
   EventSetTimer(60); // 60 seconds = 1 minute; 600 seconds = 10 minutes
   
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- destroy timer
   EventKillTimer();
   
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
//void OnTick()
//  {
//---
//   
//  }
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
  {
//---    
     GetRequest(0); 
  }
//+------------------------------------------------------------------+
//| Web Request function                                             |
//+------------------------------------------------------------------+

double GetRequest(int index // the index of the array which should be returned
                           ){
         string values[18];  
         string cookie = NULL;
    
         //string referer = NULL;
    
         int timeout = 15000; // 15 secs
    
         char data[], result[];
    
         int data_size;
    
         string headers;
    
         string url = "http://127.0.0.5/eurusd";
    
         data_size = ArraySize(data);
    
         int res; 
    
         res = WebRequest("GET", url, cookie, NULL, timeout, data, data_size, result, headers);
    
         if (res == -1) {
         Print("Error in WebRequest. Error code = ", GetLastError());
        
         } else {
            ConvertResponse(CharArrayToString(result), values);
          
            Print("p06 =>", values[5]);
            Print("p11 =>", values[10]);
                    
            return (double)values[index];      
           }
    return -1;

}


//double p06 = GetRequest(5);
//double p11 = GetRequest(10);



void ConvertResponse(string response, string& values[]) {
// parsing json object with this function...

         Print("Response =>", response);
   
         string value;
         string items[];
         ushort comma, dot;
   
         response = StringSubstr(response, 1, StringLen(response) - 2);
   
         comma = StringGetCharacter(",", 0);
         dot = StringGetCharacter(":", 0);
   
         int n = StringSplit(response, comma, items);
   
         for(int i = 0; i < n; i++) {
            string result[];
      
            StringSplit(items[i], dot, result);
      
            string key = result[0];
      
            if(key == "\"p06\"") {
               value = StringSubstr(result[1], 1, StringLen(result[1]) - 2);
            } else {
               value = result[1];
            }
      
            values[i] = value;
            
         }
}         


// Main strategy code
void OnTick() {
   // added this line and wrapped entire function within this if statement:
    if(LastActionTime != Time[0]){
    
    bool NewBar = Time[0] != LastBarTime; //original strategy code
    LastBarTime = Time[0]; // original strategy code
    
    /* try to run code only when new bar starts
    bool NewBar = LastActionTime;
    if (LastActionTime != Time[0]) {
        Print("This code is executed only once per bar", Time[0]);
        LastActionTime = Time[0];
        GetRequest(0);
    }
    
    
    from https://mql4tradingautomation.com/mql4-on-new-bar/
   
    bool NewBar = NewBarTime;
    
    //if (isNewBar()) {
      //  Print("This code is executed only once per bar");
         
    //}
    */
    
    if (IsTradeAllowed() && Bars >= MaxBarsBack) {

        ManageOrders(Symbol(), STRATORDERID, MaxSlippage);

        if (NewBar) { // tried addeding "&& isNewBar()" to original code, didn't work
            
            // Average true range
            double AveTR = iATR(NULL, 0, NATR, 1);

            // Entry and exit conditions
            /*
            double VarL1 = iCustom(NULL, 0, "p05", 0, 1);
            double VarL2 = iCustom(NULL, 0, "p01", 0, 1);
            double VarL3 = iCustom(NULL, 0, "p08", 0, Shift1 + 1); shift 10 bars back
            double VarL4 = iCustom(NULL, 0, "p05", 0, 1);
            double VarL5 = iCustom(NULL, 0, "p06", 0, Shift2 + 1); shift 3 bars back
            double VarL6 = iCustom(NULL, 0, "p11", 0, 1);
            double VarL7 = iCustom(NULL, 0, "p03", 0, Shift3 + 1); shift 18 bars back
            double VarL8 = iCustom(NULL, 0, "p04", 0, 1);
            double VarS1 = iCustom(NULL, 0, "p05", 0, 1);
            double VarS2 = iCustom(NULL, 0, "p01", 0, 1);
            double VarS3 = iCustom(NULL, 0, "p08", 0, Shift1 + 1);
            double VarS4 = iCustom(NULL, 0, "p05", 0, 1);
            double VarS5 = iCustom(NULL, 0, "p06", 0, Shift2 + 1);
            double VarS6 = iCustom(NULL, 0, "p11", 0, 1);
            double VarS7 = iCustom(NULL, 0, "p03", 0, Shift3 + 1);
            double VarS8 = iCustom(NULL, 0, "p04", 0, 1);
            */
            
            //cut number of webrequests by eliminating duplicate webrequests:
            double VarL1, VarS1; //p05
            VarL1 = VarS1 = GetRequest(4);
            double VarL2, VarS2; //p01
            VarL2 = VarS2 = GetRequest(0);
            double VarL3, VarS3; //p08s10
            VarL3 = VarS3 = GetRequest(15);
            double VarL4, VarS4; //p05
            VarL4 = VarS4 = GetRequest(4);
            double VarL5, VarS5; //p06s3
            VarL5 = VarS5 = GetRequest(16);
            double VarL6, VarS6; //p11
            VarL6 = VarS6 = GetRequest(10);
            double VarL7, VarS7; //p03s18
            VarL7 = VarS7 = GetRequest(17);
            double VarL8, VarS8; //p04
            VarL8 = VarS8 = GetRequest(3);
            
            
            /*
            double VarL1 = GetRequest(4); //p05
            double VarL2 = GetRequest(0); //p01
            double VarL3 = GetRequest(15); //p08s10
            double VarL4 = GetRequest(4); //p05
            double VarL5 = GetRequest(16); //p06s3
            double VarL6 = GetRequest(10); //p11
            double VarL7 = GetRequest(17); //p03s18
            double VarL8 = GetRequest(3); //p04
            double VarS1 = GetRequest(4); //p05
            double VarS2 = GetRequest(0); //p01
            double VarS3 = GetRequest(15); //p08s10
            double VarS4 = GetRequest(4); //p05
            double VarS5 = GetRequest(16); //p06s3
            double VarS6 = GetRequest(10); //p11
            double VarS7 = GetRequest(17); //p03s18
            double VarS8 = GetRequest(3); //p04
            
            
            double VarL1 = "p05"; //p05
            double VarL2 = "p01"; //p01
            double VarL3 = "p08s10"; //p08s10
            double VarL4 = "p05"; //p05
            double VarL5 = "p06s3"; //p06s3
            double VarL6 = "p11"; //p11
            double VarL7 = "p03s18"; //p03s18
            double VarL8 = "p04"; //p04
            double VarS1 = "p05"; //p05
            double VarS2 = "p01"; //p01
            double VarS3 = "p08s10"; //p08s10
            double VarS4 = "p05"; //p05
            double VarS5 = "p06s3"; //p06s3
            double VarS6 = "p11"; //p11
            double VarS7 = "p03s18"; //p03s18
            double VarS8 = "p04"; //p04
            
            
            Print("VarL1 = ", VarL1);
            Print("VarS1 = ", VarS1);
            Print("VarL2 = ", VarL2);
            Print("VarS2 = ", VarS2);
            Print("VarL3 = ", VarL3);
            Print("VarS3 = ", VarS3);
            Print("VarL4 = ", VarL4);
            Print("VarS4 = ", VarS4);
            Print("VarL5 = ", VarL5);
            Print("VarS5 = ", VarS5);
            Print("VarL6 = ", VarL6);
            Print("VarS6 = ", VarS6);
            Print("VarL7 = ", VarL7);
            Print("VarS7 = ", VarS7);
            Print("VarL8 = ", VarL8);
            Print("VarS8 = ", VarS8);
            */
            
            bool CondL1 = VarL1 > VarL2;
            bool CondL2 = VarL3 < VarL4;
            bool CondL3 = VarL5 < VarL6;
            bool CondL4 = VarL7 < VarL8;
            bool CondL5 = CondL3 == CondL4;
            bool CondL6 = CondL2 || CondL5;
            bool CondS1 = VarS1 < VarS2;
            bool CondS2 = VarS3 > VarS4;
            bool CondS3 = VarS5 > VarS6;
            bool CondS4 = VarS7 > VarS8;
            bool CondS5 = CondS3 == CondS4;
            bool CondS6 = CondS2 || CondS5;
            bool EntCondL = CondL1 && CondL6;
            bool EntCondS = CondS1 && CondS6;

            // Position sizing calculations
            double NShares = PSParam;

            if (RoundPS && RoundTo > 0)
                NShares = MathFloor(NShares/RoundTo) * RoundTo;

            NShares = MathMax(NShares, MinSize);
            NShares = MathMin(NShares, SizeLimit);

            double Lots = NShares/SharesPerLot;

            // Prepare to place trading orders
            int MarketPosition = CurrentPosition();
            double InitialStop = 0;

            // Entry orders
            if (MarketPosition == 0 && EntCondL) {
                EnterLongMarket(Symbol(), Lots, InitialStop, MarketPosition, MaxSlippage, STRATORDERID);
            }

            if (MarketPosition == 0 && EntCondS && !EntCondL) {
                EnterShortMarket(Symbol(), Lots, InitialStop, MarketPosition, MaxSlippage, STRATORDERID);
            }

            // Exit orders, long trades
            if (MarketPosition == 1) {
                if (iBarShift(NULL, 0, OpenEntryTime) - 1 == 0) {
                    LStop = (1 - MMStopPct/100.0) * OpenEntryPrice;
                    LStop = NormalizeDouble(LStop, Digits);
                    LTrailOn = false;
                }

                if (Close[1] - OpenEntryPrice > ATRFrTrail * AveTR)
                    LTrailOn = true;

                if (LTrailOn) {
                    NewLStop = OpenEntryPrice + TrailPct * (Close[1] - OpenEntryPrice)/100.0;
                    LStop = MathMax(LStop, NewLStop);
                    LStop = NormalizeDouble(LStop, Digits);
                }

                PlaceLongStop(Symbol(), LStop, MaxSlippage);
            }

            // Exit orders, short trades
            if (MarketPosition == -1) {
                if (iBarShift(NULL, 0, OpenEntryTime) - 1 == 0) {
                    SStop = (1 + MMStopPct/100.0) * OpenEntryPrice;
                    SStop = NormalizeDouble(SStop, Digits);
                    STrailOn = false;
                }

                if (OpenEntryPrice - Close[1] > ATRFrTrail * AveTR)
                    STrailOn = true;

                if (STrailOn) {
                    NewSStop = OpenEntryPrice - TrailPct * (OpenEntryPrice - Close[1])/100.0;
                    SStop = MathMin(SStop, NewSStop);
                    SStop = NormalizeDouble(SStop, Digits);
                }

                PlaceShortStop(Symbol(), SStop, MaxSlippage);
            }
        }
    }
    
    //Print("test bar statement",Time[0]);
               LastActionTime=Time[0];
               }
}


