//+------------------------------------------------------------------+
//|                                             R904_SS60_USDJPY.mq4 |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+

/*-------------------------------------------------------------------------------
 Trading Strategy Code
 Population member: 382
 Max bars back: 80

 Created:    7/23/2019 04:58:14

 Scripting language: MetaTrader 4 (MQL4)

 Symbol file:  .csv
 Build dates: 1/21/2016 to 1/17/2018

 Project file: .gpstrat
-------------------------------------------------------------------------------*/
#include <AdaptradeBuilderInclude.mqh>

#define STRATORDERID  19723897

// Strategy inputs
extern double X1 = -1.5798;        // Rate of change indicator value
extern int NATR = 80;              // Indicator look-back length (bars)

//original code: extern double ATRFrTrail = 5.7949; // Multiple of price difference (e.g., ATR); exit
extern double ATRFrTrail = 4.6210;

//original code: extern double TrailPct = 56.0000;  // Trailing stop percentage
extern double TrailPct = 95.000;

// original code: extern double MMStopPct = 4.661;   // Value of percentage money management stop
extern double MMStopPct = 1.971;   // Value of percentage money management stop -- testing manual override value

extern double PSParam = 100000.00; // Position sizing parameter value
extern bool RoundPS = true;        // Round-to-nearest (true/false)
extern int RoundTo = 1;            // Round-to position size value
extern int MinSize = 1;            // Minimum allowable position size
extern int SizeLimit = 100000;     // Maximum allowable position size

// Global strategy variables
int MaxBarsBack = 80;
double PointValue = 0.009270;
int MaxSlippage = 3;
double SharesPerLot = 100000;
datetime LastBarTime = Time[0]; // original strategy code
// added this:
datetime LastActionTime;

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
   EventSetTimer(60); //60 seconds = 1 minute 600 seconds = 10 minutes
   
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
         string values[15];  
         string cookie = NULL;
    
         //string referer = NULL;
    
         int timeout = 15000; // 15 secs
    
         char data[], result[];
    
         int data_size;
    
         string headers;
    
         string url = "http://127.0.0.3/usdjpy";
    
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

    bool NewBar = Time[0] != LastBarTime;
    LastBarTime = Time[0];

    if (IsTradeAllowed() && Bars >= MaxBarsBack) {

        ManageOrders(Symbol(), STRATORDERID, MaxSlippage);

        if (NewBar) {

            // Average true range
            double AveTR = iATR(NULL, 0, NATR, 1);

            // Entry and exit conditions
            /*
            double VarL1 = iCustom(NULL, 0, "p10", 0, 1);
            double VarL2 = iCustom(NULL, 0, "p09", 0, 1);
            double VarL3 = iCustom(NULL, 0, "p10", 0, 1);
            double VarL4 = iCustom(NULL, 0, "p11", 0, 1);
            double VarL5 = iCustom(NULL, 0, "p15", 0, 1);
            double VarS1 = iCustom(NULL, 0, "p10", 0, 1);
            double VarS2 = iCustom(NULL, 0, "p09", 0, 1);
            double VarS3 = iCustom(NULL, 0, "p10", 0, 1);
            double VarS4 = iCustom(NULL, 0, "p11", 0, 1);
            double VarS5 = iCustom(NULL, 0, "p15", 0, 1);
            */
            
            
            //cut number of webrequests by eliminating duplicates:
            double VarL1, VarS1; //p10
            VarL1 = VarS1 = GetRequest(9);
            double VarL2, VarS2; //p09
            VarL2 = VarS2 = GetRequest(8);
            double VarL3, VarS3; //p10
            VarL3 = VarS3 = GetRequest(9);
            double VarL4, VarS4; //p11
            VarL4 = VarS4 = GetRequest(10);
            double VarL5, VarS5; //p15
            VarL5 = VarS5 = GetRequest(14);
            
            
            /*
            double VarL1 = GetRequest(9); //p10
            double VarL2 = GetRequest(8); //p09
            double VarL3 = GetRequest(9); //p10
            double VarL4 = GetRequest(10); //p11
            double VarL5 = GetRequest(14); //p15
            double VarS1 = GetRequest(9); //p10
            double VarS2 = GetRequest(8); //p09
            double VarS3 = GetRequest(9); //p10
            double VarS4 = GetRequest(10); //p11
            double VarS5 = GetRequest(14); //p15
            
            
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
            */
            
            bool CondL1 = VarL1 > 0;
            bool CondL2 = VarL2 <= X1;
            bool CondL3 = VarL3 > 0;
            bool CondL4 = VarL4 > VarL5;
            bool CondL5 = CondL3 == CondL4;
            bool CondL6 = CondL2 || CondL5;
            bool CondS1 = VarS1 < 0;
            bool CondS2 = VarS2 >= X1;
            bool CondS3 = VarS3 < 0;
            bool CondS4 = VarS4 < VarS5;
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
