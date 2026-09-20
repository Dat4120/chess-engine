#include <iostream>
#include <string>
#include <cstdint>
#include <chrono>
#include <cctype>
#include <random>
#include <iomanip>
#include <cstring>
#include <windows.h>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <thread>
#include <vector>
#include <memory>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <new>
#ifdef _MSC_VER
    #include <malloc.h>
#endif

#define USE_NNUE true
#define CLI false
#define THREADS 1

#if defined(_MSC_VER)
    #undef CLI
    #define CLI true
#endif


#if defined(__GNUC__) || defined(__clang__)
    #define min(a,b) std::min(a,b)
    #define max(a,b) std::max(a,b)
#endif


inline int counttrailingzeros(uint64_t x){
    #if defined(_MSC_VER)
        unsigned long ind=0;
        _BitScanForward64(&ind,x);
        return ind;
    #elif defined(__GNUC__) || defined(__clang__)
        return __builtin_ctzll(x);
    #endif
}

inline int popcount(uint64_t x){
    #if defined(_MSC_VER)
        return __popcnt64(x);
    #elif defined(__GNUC__) || defined(__clang__)
        return __builtin_popcountll(x);
    #endif
}

inline int poplsb(uint64_t &x){
    #if defined(_MSC_VER)
        unsigned long ind=0;
        _BitScanForward64(&ind,x);
        x&=x-1;
        return ind;
    #elif defined(__GNUC__) || defined(__clang__)
        int ind=__builtin_ctzll(x);
        x&=x-1;
        return ind;
    #endif
}

using Bitboard=uint64_t;
using Move=uint16_t;
using Sq=int;
using Dir=int;
using Side=bool;
using Piece=int;
using Key=uint64_t;
using Value=int16_t;
using Score=int32_t;

constexpr int PAWN=0,KNIGHT=1,BISHOP=2,ROOK=3,QUEEN=4,KING=5,WHITE=1,BLACK=0;
constexpr int PAWN_W=1,KNIGHT_W=2,BISHOP_W=3,ROOK_W=4,QUEEN_W=5,KING_W=6,PAWN_B=7,KNIGHT_B=8,BISHOP_B=9,ROOK_B=10,QUEEN_B=11,KING_B=12;
constexpr int BB_indexPiece[13]={PAWN,PAWN,KNIGHT,BISHOP,ROOK,QUEEN,KING,PAWN,KNIGHT,BISHOP,ROOK,QUEEN,KING};
constexpr int BB_indexColor[13]={WHITE,WHITE,WHITE,WHITE,WHITE,WHITE,WHITE,BLACK,BLACK,BLACK,BLACK,BLACK,BLACK};

constexpr Bitboard fileA=0x0101010101010101ull;
constexpr Bitboard fileH=0x8080808080808080ull;
constexpr Bitboard not_fileA=~fileA;
constexpr Bitboard not_fileH=~fileH;

constexpr Bitboard rank1=0x00000000000000ffull;
constexpr Bitboard rank2=0x000000000000ff00ull;
constexpr Bitboard rank3=0x0000000000ff0000ull;
constexpr Bitboard rank4=0x00000000ff000000ull;
constexpr Bitboard rank5=0x000000ff00000000ull;
constexpr Bitboard rank6=0x0000ff0000000000ull;
constexpr Bitboard rank7=0x00ff000000000000ull;
constexpr Bitboard rank8=0xff00000000000000ull;

constexpr Bitboard rank456=rank4|rank5|rank6;
constexpr Bitboard rank543=rank5|rank4|rank3;
constexpr Bitboard rank23=rank2|rank3;
constexpr Bitboard rank76=rank7|rank6;

constexpr Bitboard whiteSide=rank1|rank2|rank3|rank4;
constexpr Bitboard blackSide=rank5|rank6|rank7|rank8;

constexpr Bitboard EMPTY=0;
constexpr Bitboard FULL=~EMPTY;
constexpr Bitboard LIGHT_SQUARES=0x55aa55aa55aa55aaull;
constexpr Bitboard DARK_SQUARES=~LIGHT_SQUARES;

constexpr Dir NORTH=8;
constexpr Dir SOUTH=-8;
constexpr Dir EAST=1;
constexpr Dir WEST=-1;
constexpr Dir NORTHEAST=9;
constexpr Dir NORTHWEST=7;
constexpr Dir SOUTHEAST=-7;
constexpr Dir SOUTHWEST=-9;


constexpr Bitboard shiftRight(Bitboard b){return (b<<1)&not_fileA;}
constexpr Bitboard shiftLeft(Bitboard b){return (b>>1)&not_fileH;}
constexpr Bitboard fillUp(Bitboard b){
    b|=b<<8;
    b|=b<<16;
    b|=b<<32;
    return b;
}
constexpr Bitboard fillDown(Bitboard b){
    b|=b>>8;
    b|=b>>16;
    b|=b>>32;
    return b;
}


constexpr int STATE_SIZE_ALLOC=1024;
constexpr uint32_t TT_SIZE=1<<20;
constexpr uint32_t TT_PERFT_SIZE=1<<1;


#if USE_NNUE

constexpr int HIDDEN_SIZE=512;
constexpr int INPUT_SIZE=768;
constexpr int SCALE=400;
constexpr int16_t QA=255;
constexpr int16_t QB=64;


const int OUTPUT_BUCKETS=8;
const int DIVISOR=(32+OUTPUT_BUCKETS-1)/OUTPUT_BUCKETS;

int getBucket(Bitboard occupancy){
    return (popcount(occupancy)-2)/DIVISOR;
}


struct alignas(64) Accumulator {
    int16_t vals[HIDDEN_SIZE];

    void reset(int16_t* featureBias){
        memcpy(vals,featureBias,HIDDEN_SIZE*sizeof(int16_t));
    }

    void add(const int16_t* featureCol){
        // for(int i=0;i<HIDDEN_SIZE;i++)
        //     vals[i]+=featureCol[i];

        for(int i=0;i<HIDDEN_SIZE;i+=16){
            __m256i val=_mm256_load_si256((const __m256i*)&vals[i]);
            __m256i feat=_mm256_loadu_si256((const __m256i*)&featureCol[i]);
            _mm256_store_si256((__m256i*)&vals[i],_mm256_add_epi16(val,feat));
        }
    }

    void sub(const int16_t* featureCol){
        // for(int i=0;i<HIDDEN_SIZE;i++)
        //     vals[i]-=featureCol[i];
        
        for(int i=0;i<HIDDEN_SIZE;i+=16){
            __m256i val=_mm256_load_si256((const __m256i*)&vals[i]);
            __m256i feat=_mm256_loadu_si256((const __m256i*)&featureCol[i]);
            _mm256_store_si256((__m256i*)&vals[i],_mm256_sub_epi16(val,feat));
        }
    }
};
#endif

struct UndoState{
    
    Key key;
    Key pawnKey;
    Key nonPawnKey[2];
    Key minorKey;
    Key majorKey;
    int* continuationHistory;
    int castling;
    int halfmoves;
    Sq enpassant;
    Piece moved;
    Piece captured;
    int staticEval;
    Move move;
    Move excludedMove=0;
    
    #if USE_NNUE
    Accumulator whiteAccumulator;
    Accumulator blackAccumulator;
    #endif

};


constexpr int MIDGAME=0,ENDGAME=128;

constexpr Score makeScore(Value mg,Value eg){return (static_cast<int32_t>(static_cast<uint16_t>(mg))<<16)|static_cast<uint16_t>(eg);}
#define S(mg,eg) makeScore(mg,eg)

constexpr Value mgValue(Score score){return static_cast<Value>(score>>16);}
constexpr Value egValue(Score score){return static_cast<Value>(score);}


constexpr int lerpPhase(Value mg,Value eg,uint32_t t){
    return mg+((int32_t(eg-mg)*t)>>7);
}
constexpr int lerpPhase(Score score,uint32_t t){
    return lerpPhase(mgValue(score),egValue(score),t);
}

//a mix of PESTO and handcrafted values
constexpr int PST_MID[6][64]={
    {0,0,0,0,0,0,0,0,-35,-1,-20,-23,-15,24,38,-22,-26,-4,-4,-10,3,3,33,-12,-27,-2,-5,12,17,6,10,-25,-14,13,6,21,23,12,17,-23,-6,7,26,31,65,56,25,-20,98,134,61,95,68,126,34,-11,0,0,0,0,0,0,0,0},
    {-105,-21,-58,-33,-17,-28,-19,-23,-29,-53,-12,-3,-1,18,-14,-19,-23,-9,12,10,19,17,25,-16,-13,4,16,13,28,19,21,-8,-9,17,19,53,37,69,18,22,-47,60,37,65,84,129,73,44,-73,-41,72,36,23,62,7,-17,-167,-89,-34,-49,61,-97,-15,-107},
    {-33,-3,-14,-21,-13,-12,-39,-21,4,15,16,0,7,21,33,1,0,15,15,15,14,27,18,10,-6,13,13,26,34,12,10,4,-4,5,19,50,37,37,7,-2,-16,37,43,40,35,50,37,-2,-26,16,-18,-13,30,59,18,-47,-29,4,-82,-37,-25,-42,7,-8},
    {-19,-13,1,17,16,7,-37,-26,-44,-16,-20,-9,-1,11,-6,-71,-45,-25,-16,-17,3,0,-5,-33,-36,-26,-12,-1,9,-7,6,-23,-24,-11,7,26,24,35,-8,-20,-5,19,26,36,17,45,61,16,27,32,58,62,80,67,26,44,32,42,32,51,63,9,31,43},
    {-1,-18,-9,10,-15,-25,-31,-50,-35,-8,11,2,8,15,-3,1,-14,2,-11,-2,-5,2,14,5,-9,-26,-9,-10,-2,-4,3,-3,-27,-27,-16,-16,-1,17,-2,1,-13,-17,7,8,29,56,47,57,-24,-39,-5,1,-16,57,28,54,-28,0,29,12,59,44,43,45},
    {-15,36,12,-54,8,-28,24,14,1,7,-8,-64,-43,-16,9,8,-14,-14,-22,-46,-44,-30,-15,-27,-49,-1,-27,-39,-46,-44,-33,-51,-17,-20,-12,-27,-30,-25,-14,-36,-9,24,2,-16,-20,6,22,-22,29,-1,-20,-7,-8,-4,-38,-29,-65,23,16,-15,-56,-34,2,13}
};
constexpr int PST_END[6][64]={
    {0,0,0,0,0,0,0,0,13,8,8,10,13,0,2,-7,4,7,-6,1,0,-5,-1,-8,13,9,-3,-7,-7,-8,3,-1,32,24,13,5,-2,4,17,17,94,100,85,67,56,53,82,84,178,173,158,134,147,132,165,187,0,0,0,0,0,0,0,0},
    {-29,-51,-23,-15,-22,-18,-50,-64,-42,-20,-10,-5,-2,-20,-23,-44,-23,-3,-1,15,10,-3,-20,-22,-18,-6,16,25,16,17,4,-18,-17,3,22,22,22,11,8,-18,-24,-20,10,9,-1,-9,-19,-41,-25,-8,-25,-2,-9,-25,-24,-52,-58,-38,-13,-28,-31,-27,-63,-99},
    {-23,-9,-23,-5,-9,-16,-5,-17,-14,-18,-7,-1,4,-9,-15,-27,-12,-3,8,10,13,3,-7,-15,-6,3,13,19,7,10,-3,-9,-3,9,12,9,14,10,3,2,2,-8,0,-1,-2,6,0,4,-8,-4,7,-12,-3,-13,-4,-14,-14,-21,-11,-8,-7,-9,-17,-24},
    {-9,2,3,-1,-5,-13,4,-20,-6,-6,0,2,-9,-9,-11,-3,-4,0,-5,-1,-7,-12,-8,-16,3,5,8,4,-5,-6,-8,-11,4,3,13,1,2,1,-1,2,7,7,7,5,4,-3,-5,-3,11,13,13,11,-3,3,8,3,13,10,18,15,12,12,8,5},
    {-33,-28,-22,-43,-5,-32,-20,-41,-22,-23,-30,-16,-16,-23,-36,-32,-16,-27,15,6,9,17,10,5,-18,28,19,47,31,34,39,23,3,22,24,45,57,40,57,36,-20,6,9,49,47,35,19,9,-17,20,32,41,58,25,30,0,-9,22,22,27,27,19,10,20},
    {-53,-34,-21,-11,-28,-14,-24,-43,-27,-11,4,13,14,4,-5,-17,-19,-3,11,21,23,16,7,-9,-18,-4,21,24,27,23,9,-11,-8,22,24,27,26,33,26,3,10,17,23,15,20,45,44,13,-12,17,14,17,17,38,23,11,-74,-35,-18,-18,-11,15,4,-17}
};

static Score pst[12][64];

constexpr static float M=1;
constexpr static Score pieceValueScores[5]={S(100*M,130*M),S(300*M,300*M),S(315*M,320*M),S(500*M,510*M),S(900*M,900*M)};

/*mvvlva table hardcoded from:
int MVVLVA[13][13];
int value[13]={0,1,2,3,4,5,0,1,2,3,4,5,0};
int main(){
    for(int v=0;v<13;v++){
        std::cout<<"{";
        for(int a=0;a<13;a++){
            MVVLVA[v][a]=(value[v]*10-value[a])+5;
            if(v==0||a==0)MVVLVA[v][a]=0;
            std::cout<<MVVLVA[v][a]<<(a==12?"":","); 
        }
        std::cout<<"},\n";
    }
    return 0;
}*/
constexpr static int MVVLVA[13][13]={
    {0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,14,13,12,11,10,15,14,13,12,11,10,15},
    {0,24,23,22,21,20,25,24,23,22,21,20,25},
    {0,34,33,32,31,30,35,34,33,32,31,30,35},
    {0,44,43,42,41,40,45,44,43,42,41,40,45},
    {0,54,53,52,51,50,55,54,53,52,51,50,55},
    {0,4,3,2,1,0,5,4,3,2,1,0,5},
    {0,14,13,12,11,10,15,14,13,12,11,10,15},
    {0,24,23,22,21,20,25,24,23,22,21,20,25},
    {0,34,33,32,31,30,35,34,33,32,31,30,35},
    {0,44,43,42,41,40,45,44,43,42,41,40,45},
    {0,54,53,52,51,50,55,54,53,52,51,50,55},
    {0,4,3,2,1,0,5,4,3,2,1,0,5}
};

static int lmrTable[64][64];

constexpr int CORR_HIST_SIZE=16384;
constexpr int corrHistSizeAnd=CORR_HIST_SIZE-1;

// a move is 16 bits: from 0-5, to 6-11, promotion type 12-13, flag 14-15
//promotion type: 0 knight, 1 bishop, 2 rook, 3 queen
//flag: 1 promotion, 2 en passant, 3 castling
constexpr int FLAG_PROMOTION=0b01,FLAG_EN_PASSANT=0b10,FLAG_CASTLING=0b11;

constexpr Move packMove(Sq from,Sq to){return from|(to<<6);}
constexpr Move packMove_promotion(Sq from,Sq to,int promotion){return from|(to<<6)|(promotion<<12)|(FLAG_PROMOTION<<14);}
constexpr Move packMove_flag(Sq from,Sq to,int flag){return from|(to<<6)|(flag<<14);};

static Bitboard betweens[64][64];
static Bitboard rays[64][64];
static int distances[64][64];
static int centerDistances[64]={
    3, 3, 3, 3, 3, 3, 3, 3,
    3, 2, 2, 2, 2, 2, 2, 3,
    3, 2, 1, 1, 1, 1, 2, 3,
    3, 2, 1, 0, 0, 1, 2, 3,
    3, 2, 1, 0, 0, 1, 2, 3,
    3, 2, 1, 1, 1, 1, 2, 3,
    3, 2, 2, 2, 2, 2, 2, 3,
    3, 3, 3, 3, 3, 3, 3, 3
};

static int castlingRights[64];
static Bitboard knightAttacks[64];
static Bitboard kingAttacks[64];

static Bitboard rookMasks[64];
static Bitboard bishopMasks[64];

static uint64_t rookMagics[64]={36031546090078208ull,90089587954688005ull,2341881152618369024ull,144119625049833504ull,2341880602359496832ull,2450046166809774084ull,36029896564146304ull,72057871066529794ull,18718087025033250ull,576601627234926724ull,563095983374472ull,173529391862120576ull,281552370010112ull,562958543881232ull,1243837926379487236ull,19703250534137940ull,107202387902592ull,141837608160512ull,634418746626304ull,3711257463803093251ull,1369236673610648576ull,1225120385922499584ull,576465151994171400ull,621498947611934980ull,459405105880416256ull,4503884168962052ull,35186520621185ull,563233422319624ull,41658315888920576ull,738594739091145216ull,2308182924576530952ull,9232978495717016660ull,36028934495666384ull,11529390984173457408ull,324277865958875136ull,5769111157029736580ull,4512397876265984ull,2882308161719501312ull,9009402640091184ull,18164209149817861ull,1197992960138969088ull,864762598892109856ull,2603080653473857600ull,2314867800788729984ull,1134696285143044ull,9656282750126162048ull,576461851848638592ull,2269669042552844ull,457673862611712ull,150589251054109952ull,585476816911794432ull,144396731772576000ull,20266234897531264ull,1198520468282802688ull,1152922612981564416ull,4971992697635111424ull,1442313514861480193ull,18032540992405761ull,1135804638302273ull,110352110870106113ull,1171498992791454850ull,563001816518658ull,5188164367350268420ull,2305845355540939010ull};
static int rookShifts[64]={52,53,53,53,53,53,53,52,53,54,54,54,54,54,54,53,53,54,54,54,54,54,54,53,53,54,54,54,54,54,54,53,53,54,54,54,54,54,54,53,53,54,54,54,54,54,54,53,53,54,54,54,54,54,54,53,52,53,53,53,53,53,53,52};
static uint64_t bishopMagics[64]={455620352916602944ull,7016644580981408768ull,9802086805747793922ull,45212521610629376ull,280353612297672768ull,288794494474651648ull,153267591719422976ull,72198623728781316ull,74353511827263554ull,4742299409884676354ull,4521208998019077ull,2260638858477638ull,9020415406014594ull,1171503530827056256ull,309372129410ull,72199448226432097ull,20275131990411457ull,326792456820232338ull,72620612912693256ull,290482178146451488ull,36310450509447180ull,9226186860761448706ull,281492174508032ull,9223515115167482112ull,4622949450474586665ull,6920379787118904368ull,2883517622895870336ull,40536795838152712ull,4648735170633728ull,9227882302892671488ull,3457160927580162ull,6920344053965985800ull,571883653439488ull,10386427880795276320ull,18058396154988576ull,9225768989383721248ull,1153204096301662240ull,1738398806345547856ull,13687983479734784ull,5046883870974017664ull,143074491633696ull,289686138170580993ull,9043492806420480ull,275150873088ull,9161232353600ull,2927344191414485504ull,81066447476097537ull,4920200737604281856ull,2891383538475597952ull,1450449660392249408ull,220247064842752ull,45071206583502912ull,29273536124453152ull,351852378525700ull,2306986647369291790ull,369321562035268676ull,11402118150766592ull,144169068579062912ull,1152961125953900552ull,9229098295562471424ull,288231063883482240ull,72092847185068289ull,576887381203353890ull,9008316298826016ull};
static int bishopShifts[64]={58,59,59,59,59,59,59,58,59,59,59,59,59,59,59,59,59,59,57,57,57,57,59,59,59,59,57,55,55,57,59,59,59,59,57,55,55,57,59,59,59,59,57,57,57,57,59,59,59,59,59,59,59,59,59,59,58,59,59,59,59,59,59,58};

//hardcoded table sizes after computing magics
static Bitboard rookTable[102400];
static Bitboard bishopTable[5248];

static int rookOffsets[64];
static int bishopOffsets[64];

int rookOffsetInc=0;
int bishopOffsetInc=0;

static Key zobristPieceSq[12][64];
//file of en passant, if capture is possible
static Key zobristEnpassant[8];
//position castling state is 4 bits, describing white/black's king/queen side rights
static Key zobristCastling[16];
//active in the key if it is white's turn
static Key zobristSide;

uint64_t random64(){
    static std::mt19937_64 rng(1312026);
    return rng();
}

void addMoveRay(Bitboard &moves,Sq sq,int x,int y,int dx,int dy,Bitboard blockers,bool excludeEdges){
    
    int sx=0,sy=0,ex=8,ey=8;

    if(excludeEdges){
        if(dx==1)ex=7;
        if(dx==-1)sx=1;
        if(dy==1)ey=7;
        if(dy==-1)sy=1;
    }

    for(int i=1;i<8;i++){
        int _x=x+dx*i,_y=y+dy*i,_sq=(_y<<3)+_x;
        if(_x>=sx&&_x<ex&&_y>=sy&&_y<ey){
            moves|=1ull<<_sq;
            if(blockers&1ull<<_sq)break;
        }else
            break;
    }
}

Bitboard generateSliderMoves(Sq sq,Bitboard blockers,bool usingRook,bool excludeEdges){
    
    Bitboard moves=0;
    int x=sq&7,y=sq>>3;
    if(usingRook){
        addMoveRay(moves,sq,x,y,0,1,blockers,excludeEdges);
        addMoveRay(moves,sq,x,y,0,-1,blockers,excludeEdges);
        addMoveRay(moves,sq,x,y,-1,0,blockers,excludeEdges);
        addMoveRay(moves,sq,x,y,1,0,blockers,excludeEdges);
    }else{
        addMoveRay(moves,sq,x,y,1,1,blockers,excludeEdges);
        addMoveRay(moves,sq,x,y,1,-1,blockers,excludeEdges);
        addMoveRay(moves,sq,x,y,-1,1,blockers,excludeEdges);
        addMoveRay(moves,sq,x,y,-1,-1,blockers,excludeEdges);
    }
    return moves;
}
void addAttack(Bitboard& attacks,int x,int y){
    if(x>=0&&x<8&&y>=0&&y<8)
        attacks|=1ull<<((y<<3)+x);
}
void setBetweenRays(Sq sq,int x,int y,int dx,int dy){
    Bitboard currentLine=0;
    int range=1;

    for(int i=1;i<8;i++){

        int _x=x+dx*i,_y=y+dy*i,_sq=(_y<<3)+_x;

        if(_x>=0&&_x<8&&_y>=0&&_y<8){
            betweens[sq][_sq]=currentLine;
            currentLine|=1ull<<_sq;
            range++;
        }else
            break;
    }

    for(int i=1;i<range;i++){
        int _x=x+dx*i,_y=y+dy*i;
        rays[sq][(_y<<3)+_x]=currentLine;
    }
}

void initEngineTables(){
    for(int y=0;y<8;y++){
        for(int x=0;x<8;x++){
            Sq sq=(y<<3)+x;

            addAttack(knightAttacks[sq],x-2,y+1);
            addAttack(knightAttacks[sq],x-1,y+2);
            addAttack(knightAttacks[sq],x+1,y+2);
            addAttack(knightAttacks[sq],x+2,y+1);
            addAttack(knightAttacks[sq],x+2,y-1);
            addAttack(knightAttacks[sq],x+1,y-2);
            addAttack(knightAttacks[sq],x-1,y-2);
            addAttack(knightAttacks[sq],x-2,y-1);

            addAttack(kingAttacks[sq],x-1,y-1);
            addAttack(kingAttacks[sq],x,y-1);
            addAttack(kingAttacks[sq],x+1,y-1);
            addAttack(kingAttacks[sq],x+1,y);
            addAttack(kingAttacks[sq],x+1,y+1);
            addAttack(kingAttacks[sq],x,y+1);
            addAttack(kingAttacks[sq],x-1,y+1);
            addAttack(kingAttacks[sq],x-1,y);

            rookMasks[sq]=generateSliderMoves(sq,0,true,true);
            bishopMasks[sq]=generateSliderMoves(sq,0,false,true);

            castlingRights[sq]=0b1111;

            for(int i=0;i<12;i++)
                zobristPieceSq[i][sq]=random64();
        }
    }

    castlingRights[0]=0b1011;
    castlingRights[7]=0b0111;
    castlingRights[56]=0b1110;
    castlingRights[63]=0b1101;
    castlingRights[4]=0b0011;
    castlingRights[60]=0b1100;

    for(int i=0;i<8;i++)
        zobristEnpassant[i]=random64();
    for(int i=0;i<16;i++)
        zobristCastling[i]=random64();
    zobristSide=random64();

    for(Sq i=0;i<64;i++){
        int x=i&7,y=i>>3;
        
        setBetweenRays(i,x,y,1,0);
        setBetweenRays(i,x,y,-1,0);
        setBetweenRays(i,x,y,0,1);
        setBetweenRays(i,x,y,0,-1);
        setBetweenRays(i,x,y,1,1);
        setBetweenRays(i,x,y,1,-1);
        setBetweenRays(i,x,y,-1,1);
        setBetweenRays(i,x,y,-1,-1);

        for(int j=i+1;j<64;j++){
            int dx=j&7,dy=j>>3;
            distances[i][j]=max(abs(dx-x),abs(dy-y));
            distances[j][i]=distances[i][j];
        }
    }

    for(int i=0;i<6;i++){
        Score score=i<5?pieceValueScores[i]:0;

        for(Sq sq=0;sq<64;sq++){

            int rank=sq>>3,file=sq&7;
            Sq mirroredSq=((7-rank)<<3)+file;

            pst[i][sq]=score+makeScore(PST_MID[i][sq],PST_END[i][sq]);
            pst[i+6][sq]=score+makeScore(PST_MID[i][mirroredSq],PST_END[i][mirroredSq]);
        }
    }

    for(int depth=1;depth<64;depth++)
        for(int move=1;move<64;move++)
            lmrTable[depth][move]=int(0.7844+std::log(depth)*std::log(move)/2.1);
}

uint64_t setOccupancy(int index,int bits,Bitboard mask){
    Bitboard occupancy=0ull;

    for(int i=0;i<bits;i++){
        int square=counttrailingzeros(mask);
        mask&=mask-1;

        if(index&(1<<i))
            occupancy|=(1ull<<square);
    }

    return occupancy;
}
void buildMagicAttackTable(bool usingRook){

    for(Sq sq=0;sq<64;sq++){
        
        Bitboard mask=usingRook?rookMasks[sq]:bishopMasks[sq];
        uint64_t magic=usingRook?rookMagics[sq]:bishopMagics[sq];
        int shift=usingRook?rookShifts[sq]:bishopShifts[sq];
        
        int bits=popcount(mask);
        int size=1<<bits;

        if(usingRook){
            rookOffsets[sq]=rookOffsetInc;
            rookOffsetInc+=size;
        }else{
            bishopOffsets[sq]=bishopOffsetInc;
            bishopOffsetInc+=size;
        }

        for(int i=0;i<size;i++){

            Bitboard occupancy=setOccupancy(i,bits,mask);
            Bitboard attacks=usingRook?generateSliderMoves(sq,occupancy,true,false):generateSliderMoves(sq,occupancy,false,false);

            int index=(occupancy*magic)>>shift;

            if(usingRook)
                rookTable[rookOffsets[sq]+index]=attacks;
            else
                bishopTable[bishopOffsets[sq]+index]=attacks;
        }
    }
}

Bitboard getBishopAttacks(Sq sq,Bitboard blockers){
    return bishopTable[bishopOffsets[sq]+(((blockers&bishopMasks[sq])*bishopMagics[sq])>>bishopShifts[sq])];
}
Bitboard getRookAttacks(Sq sq,Bitboard blockers){
    return rookTable[rookOffsets[sq]+(((blockers&rookMasks[sq])*rookMagics[sq])>>rookShifts[sq])];
}


struct Position{

    Key key;
    Key pawnKey;
    Key nonPawnKey[2];
    Key minorKey;
    Key majorKey;
    //the zobrist keys of previous game positions before the root node of a search, used for repetition detection. works in tandem with the keys already stored in the undo stack, cuz the undo stack doesnt store keys behind the root of the search
    //only about 100 previous positions are really needed. indexed starting from 0=root, 1=the position before root, etc
    Key pastRepetitionKeys[128];

    Bitboard pieces[6];
    Bitboard colors[2];

    int board[64];

    Side turn;
    int castling;//0b0000: white kingside,white queenside,black kingside,black queenside
    Sq enpassant;//en passant sq
    int halfmoves;
    int ply=0;
    int pastRepetitionLength=0;
    
    #if USE_NNUE
        Accumulator whiteAccumulator;
        Accumulator blackAccumulator;
    #endif
    
    bool mirrorWhite;
    bool mirrorBlack;
};

#if USE_NNUE
void addToNNUE(Position *pos,Piece piece,Sq sq);
void subFromNNUE(Position *pos,Piece piece,Sq sq);
void refreshWhiteAccumulator(Position *pos);
void refreshBlackAccumulator(Position *pos);
void resetNNUE(Position *pos);
#endif

std::atomic<uint64_t> globalAllotedTime;
std::atomic<bool> globalStopFlag;

class Worker{
public:

    //silence MSVC warning
    #ifdef _MSC_VER
        static void* operator new(size_t size){
            void* ptr=_aligned_malloc(size, 64);
            if(!ptr) throw std::bad_alloc();
            return ptr;
        }
        static void operator delete(void* ptr){
            _aligned_free(ptr);
        }
    #endif

    int id;

    Position pos;

    Move moveList[64*STATE_SIZE_ALLOC];
    int moveScores[64*STATE_SIZE_ALLOC];
    Move* movePtr=moveList;

    UndoState undoStack[STATE_SIZE_ALLOC];
    UndoState* undoPtr=undoStack;

    int historyTable[12*64];
    int captureHistoryTable[12*64][6];
    int continuationHistory[12*64][12*64];
    Move killerMoves[2*STATE_SIZE_ALLOC];
    Move countermoveHistory[12*64];
    int pawnCorrectionHistory[2][CORR_HIST_SIZE];
    int nonpawnCorrectionHistory[2][2][CORR_HIST_SIZE];
    int minorCorrectionHistory[2][CORR_HIST_SIZE];
    int majorCorrectionHistory[2][CORR_HIST_SIZE];

    uint64_t nodeCount=0;
    uint64_t searchTimeLimit=0;
    bool searchTimeStopped=false;

    std::atomic<uint64_t> out_nodes;
    std::atomic<int> out_bestScore;
    std::atomic<Move> out_bestMove;

    void initSearch(Position *pos);
    void clearHistories();
    uint64_t perft(Position *pos,int depth,int startDepth);
    int scoreMove(Position *pos,Move move,Move countermove);
    void scoreMoves(Position *pos,Move* startMoveList,Move* endMoveList,Move ttMove);
    void pickMove(Move* move,Move* endMoveList);
    int qsearch(Position *pos,int alpha,int beta,bool pvNode);
    int search(Position *pos,int depth,int alpha,int beta);
    void startSearch();
};
std::vector<std::unique_ptr<Worker>>* allWorkers;

std::string uint64ToHex(uint64_t n){
    std::string out="";
    for(int i=0;i<16;i++){
        out="0123456789ABCDEF"[n&15]+out;
        n>>=4;
    }
    return out;
}
std::string getFen(Position *pos){
    std::string s="";

    for(int rank=7;rank>=0;rank--){
        int empty=0;
        for(int file=0;file<8;file++){
            Sq sq=(rank<<3)+file;
            if(pos->board[sq]==0)empty++;
            else if(empty>0){
                s+=empty+'0';
                empty=0;
            }
            if(pos->board[sq])s+="PNBRQKpnbrqk"[pos->board[sq]-1];
        }
        if(empty)s+=empty+'0';
        if(rank>0)
            s+='/';
    }

    s+=" ";
    s+=pos->turn?'w':'b';

    s+=" ";
    if(pos->castling){
        if(pos->castling&0b1000)s+="K";
        if(pos->castling&0b0100)s+="Q";
        if(pos->castling&0b0010)s+="k";
        if(pos->castling&0b0001)s+="q";
    }else{
        s+='-';
    }
    
    s+=" ";
    if(pos->enpassant==-1)
        s+="-";
    else{
        s+='a'+(pos->enpassant&7);
        s+=(pos->enpassant>>3)+'1';
    }

    s+=" "+std::to_string(pos->halfmoves);
    
    //placeholder: not available
    s+=" 1";

    return s;
}
void printPosition(Position *pos){

    std::string out="";

    out+="castling: ";
    out+=pos->castling&0b1000?"K":"-";
    out+=pos->castling&0b0100?"Q":"-";
    out+=pos->castling&0b0010?"k":"-";
    out+=pos->castling&0b0001?"q":"-";
    out+="\n";
    out+="en passant: ";
    if(pos->enpassant==-1)
        out+="-";
    else{
        out+='a'+(pos->enpassant&7);
        out+=(pos->enpassant>>3)+'1';
    }
    out+="\nhalfmoves: ";
    out+=std::to_string(pos->halfmoves);
    out+="\nkey: ";
    out+=uint64ToHex(pos->key);
    out+="  pawn: ";
    out+=uint64ToHex(pos->pawnKey);
    out+="  nonpawn: [";
    out+=uint64ToHex(pos->nonPawnKey[0]);
    out+=',';
    out+=uint64ToHex(pos->nonPawnKey[1]);
    out+="]  minor: ";
    out+=uint64ToHex(pos->minorKey);
    out+="  major: ";
    out+=uint64ToHex(pos->majorKey);
    out+="\nrep keys:";
    for(int i=0;i<pos->pastRepetitionLength;i++)
        out+=" "+uint64ToHex(pos->pastRepetitionKeys[i]);
    out+="\nrep key len: ";
    out+=std::to_string(pos->pastRepetitionLength);
    out+="\n\n";
    out+=getFen(pos);
    out+="\n\n";

    for(int y=8;y--;){
        out+='1'+y;
        out+="   ";

        for(int x=0;x<8;x++){

            Bitboard sq=1ull<<(y*8+x);

            char p='.';
            if(pos->pieces[PAWN]&sq)p='P';
            if(pos->pieces[KNIGHT]&sq)p='N';
            if(pos->pieces[BISHOP]&sq)p='B';
            if(pos->pieces[ROOK]&sq)p='R';
            if(pos->pieces[QUEEN]&sq)p='Q';
            if(pos->pieces[KING]&sq)p='K';

            if(pos->colors[BLACK]&sq){
                // out+="\033[30m";
                p=std::tolower(p);
            }else if(!(pos->colors[WHITE]&sq)){
                // out+="\033[38;5;232m";
            }


            //print pos->board instead
            // p=' ';
            // out+='0'+pos->board[y*8+x];
            
            out+=p;
            // out+="\033[0m ";
            out+=" ";
        }
        out+="\n";
    }
    
    std::cout<<(out+"\n\033[0m"+(pos->turn?"W":"B")+"   a b c d e f g h\n\n");
}

void printBitboard(Bitboard board){

    std::string out="0b";

    for(int i=64;i--;)
        out+=board&(1ull<<i)?"1":"0";
    
    out+="ull\n";

    for(int y=8;y--;){
        out+='1'+y;
        out+="   ";
        for(int x=0;x<8;x++){
            char p=board&(1ull<<(y*8+x))?'X':'.';
            out+=p;
            out+=" ";
        }
        out+="\n";
    }
    
    std::cout<<(out+"\n    a b c d e f g h\n\n");
}

constexpr int INF=100000;
constexpr int MATE_THRESHOLD=INF-1000;

struct TTEntry_perft{

    Key key=0;
    uint64_t count=0;
    int depth;
};
struct TTEntry{

    Key key=0;
    int depth;
    int score;
    int flag;
    int staticEval=INF;
    Move bestMove;
    uint16_t age;
};

uint16_t ttAge=0;

constexpr int EXACT=1,LOWER=2,UPPER=3;

static TTEntry TT[TT_SIZE];
static TTEntry_perft TT_perft[TT_PERFT_SIZE];
constexpr int TTSizeAnd=TT_SIZE-1;
constexpr int TTPerftSizeAnd=TT_PERFT_SIZE-1;

int hashfull(){
    int count=0;
    for(int i=0;i<1000;i++)
        if(TT[i].key || TT[i].bestMove)
            count++;
    return count;
}


bool sqAttacked(Position *pos,Sq sq,int side,Bitboard blockerFlip=0,Bitboard epPawnSqRemove=FULL){
    
    Bitboard sqMask=1ull<<sq;
    Bitboard color=pos->colors[side];

    //pawns
    Bitboard pawns=(pos->pieces[PAWN]&color)&epPawnSqRemove;
    Bitboard capEast=side?(sqMask&not_fileH)>>7:(sqMask&not_fileH)<<9;
    Bitboard capWest=side?(sqMask&not_fileA)>>9:(sqMask&not_fileA)<<7;
    if((capEast|capWest)&pawns)
        return true;
    
    //knights
    Bitboard knights=pos->pieces[KNIGHT]&color;
    if(knightAttacks[sq]&knights)
        return true;

    //sliders
    Bitboard bishops=pos->pieces[BISHOP]&color;
    Bitboard rooks=pos->pieces[ROOK]&color;
    Bitboard queens=pos->pieces[QUEEN]&color;
    Bitboard blockers=(pos->colors[WHITE]|pos->colors[BLACK])^blockerFlip;

    if(getBishopAttacks(sq,blockers)&(bishops|queens))
        return true;
    if(getRookAttacks(sq,blockers)&(rooks|queens))
        return true;

    //kings
    Bitboard kings=pos->pieces[KING]&color;
    if(kingAttacks[sq]&kings)
        return true;
    
    return false;
}

Bitboard sqAttackers(Position *pos,Sq sq,int side){
    
    Bitboard attackers;
    
    Bitboard sqMask=1ull<<sq;
    Bitboard color=pos->colors[side];

    //pawns
    Bitboard pawns=pos->pieces[PAWN]&color;
    Bitboard capEast=side?(sqMask&not_fileH)>>7:(sqMask&not_fileH)<<9;
    Bitboard capWest=side?(sqMask&not_fileA)>>9:(sqMask&not_fileA)<<7;
    attackers=(capEast|capWest)&pawns;
    
    //knights
    Bitboard knights=pos->pieces[KNIGHT]&color;
    attackers|=knightAttacks[sq]&knights;

    //sliders
    Bitboard bishops=pos->pieces[BISHOP]&color;
    Bitboard rooks=pos->pieces[ROOK]&color;
    Bitboard queens=pos->pieces[QUEEN]&color;
    Bitboard blockers=pos->colors[WHITE]|pos->colors[BLACK];

    attackers|=
        getBishopAttacks(sq,blockers)&(bishops|queens)
        |
        getRookAttacks(sq,blockers)&(rooks|queens);

    //kings
    Bitboard kings=pos->pieces[KING]&color;
    attackers|=kingAttacks[sq]&kings;
    
    return attackers;
}

//overload: this version finds attackers of a sq for both colors (used in SEE)
Bitboard sqAttackers(Position *pos,Sq sq){
    
    Bitboard attackers;
    
    Bitboard sqMask=1ull<<sq;
    Bitboard white=pos->colors[WHITE];
    Bitboard black=pos->colors[BLACK];

    //pawns
    Bitboard pawns=pos->pieces[PAWN];
    Bitboard capEastW=(sqMask&not_fileH)>>7;
    Bitboard capWestW=(sqMask&not_fileA)>>9;
    Bitboard capEastB=(sqMask&not_fileH)<<9;
    Bitboard capWestB=(sqMask&not_fileA)<<7;
    attackers=(capEastW|capWestW)&(pawns&white);
    attackers|=(capEastB|capWestB)&(pawns&black);

    //knights
    attackers|=knightAttacks[sq]&pos->pieces[KNIGHT];

    //sliders
    Bitboard bishops=pos->pieces[BISHOP];
    Bitboard rooks=pos->pieces[ROOK];
    Bitboard queens=pos->pieces[QUEEN];
    Bitboard blockers=white|black;

    attackers|=
        getBishopAttacks(sq,blockers)&(bishops|queens)
        |
        getRookAttacks(sq,blockers)&(rooks|queens);

    //kings
    attackers|=kingAttacks[sq]&pos->pieces[KING];
    
    return attackers;
}


void generatePseudoLegalMoves(Move*& movePtr,Position *pos,Bitboard evasionMask=FULL){

    Side side=pos->turn;

    Bitboard white=pos->colors[WHITE];
    Bitboard black=pos->colors[BLACK];

    Bitboard friendly=side?white:black;
    Bitboard not_friendly=~friendly;

    Bitboard enemy=side?black:white;
    Bitboard all=white|black;
    Bitboard empty=~all;

    Bitboard pawns=pos->pieces[PAWN]&friendly;

    Bitboard pawnPush=(side?pawns<<8:pawns>>8)&empty;
    Bitboard doublePawnPush=(side?(pawnPush<<8)&rank4:(pawnPush>>8)&rank5)&empty;

    pawnPush&=evasionMask;
    doublePawnPush&=evasionMask;

    //during check, gen all EP moves anyways
    Bitboard passant=pos->enpassant==-1?0:1ull<<pos->enpassant;
    Bitboard pawnCapEast=(side?(pawns&not_fileH)<<9:(pawns&not_fileH)>>7)&((enemy&evasionMask)|passant);
    Bitboard pawnCapWest=(side?(pawns&not_fileA)<<7:(pawns&not_fileA)>>9)&((enemy&evasionMask)|passant);

    Dir up=side?NORTH:SOUTH;
    Dir upwest=side?NORTHWEST:SOUTHWEST;
    Dir upeast=side?NORTHEAST:SOUTHEAST;

    while(pawnPush){
        Sq sq=poplsb(pawnPush);
        if(sq<8||sq>55){
            *movePtr++=packMove_promotion(sq-up,sq,3);
            *movePtr++=packMove_promotion(sq-up,sq,2);
            *movePtr++=packMove_promotion(sq-up,sq,1);
            *movePtr++=packMove_promotion(sq-up,sq,0);
        }else
            *movePtr++=packMove(sq-up,sq);
    }
    while(doublePawnPush){
        Sq sq=poplsb(doublePawnPush);
        *movePtr++=packMove(sq-up-up,sq);
    }
    while(pawnCapEast){
        Sq sq=poplsb(pawnCapEast);
        if(sq<8||sq>55){
            *movePtr++=packMove_promotion(sq-upeast,sq,3);
            *movePtr++=packMove_promotion(sq-upeast,sq,2);
            *movePtr++=packMove_promotion(sq-upeast,sq,1);
            *movePtr++=packMove_promotion(sq-upeast,sq,0);
        }else if(pos->enpassant==sq)
            *movePtr++=packMove_flag(sq-upeast,sq,FLAG_EN_PASSANT);
        else
            *movePtr++=packMove(sq-upeast,sq);
    }
    while(pawnCapWest){
        Sq sq=poplsb(pawnCapWest);
        if(sq<8||sq>55){
            *movePtr++=packMove_promotion(sq-upwest,sq,3);
            *movePtr++=packMove_promotion(sq-upwest,sq,2);
            *movePtr++=packMove_promotion(sq-upwest,sq,1);
            *movePtr++=packMove_promotion(sq-upwest,sq,0);
        }else if(pos->enpassant==sq)
            *movePtr++=packMove_flag(sq-upwest,sq,FLAG_EN_PASSANT);
        else
            *movePtr++=packMove(sq-upwest,sq);
    }

    Bitboard knights=pos->pieces[KNIGHT]&friendly;
    while(knights){
        Sq sq=poplsb(knights);
        Bitboard attacks=knightAttacks[sq]&not_friendly&evasionMask;

        while(attacks){
            Sq to=poplsb(attacks);
            *movePtr++=packMove(sq,to);
        }
    }

    Bitboard bishops=pos->pieces[BISHOP]&friendly;
    while(bishops){
        Sq sq=poplsb(bishops);
        Bitboard attacks=getBishopAttacks(sq,all)&not_friendly&evasionMask;
        
        while(attacks){
            Sq to=poplsb(attacks);
            *movePtr++=packMove(sq,to);
        }
    }

    Bitboard rooks=pos->pieces[ROOK]&friendly;
    while(rooks){
        Sq sq=poplsb(rooks);
        Bitboard attacks=getRookAttacks(sq,all)&not_friendly&evasionMask;
        
        while(attacks){
            Sq to=poplsb(attacks);
            *movePtr++=packMove(sq,to);
        }
    }

    Bitboard queens=pos->pieces[QUEEN]&friendly;
    while(queens){
        Sq sq=poplsb(queens);
        Bitboard attacks=(getBishopAttacks(sq,all)|getRookAttacks(sq,all))&not_friendly&evasionMask;
        
        while(attacks){
            Sq to=poplsb(attacks);
            *movePtr++=packMove(sq,to);
        }
    }

    Sq kingSq=counttrailingzeros(pos->pieces[KING]&friendly);
    Bitboard attacks=kingAttacks[kingSq]&not_friendly;

    while(attacks){
        Sq to=poplsb(attacks);
        *movePtr++=packMove(kingSq,to);
    }

    //if the evasion mask is full, it means not in any check rn
    if(evasionMask==FULL){
        
        //check for castling rights and empty squares for castle
        int ourCastlingMask=pos->castling&(side?0b1100:0b0011);
        //kingside
        if(ourCastlingMask&0b1010 && !(all&(0b110ull<<kingSq)))
            *movePtr++=packMove_flag(kingSq,kingSq+2,FLAG_CASTLING);
        //queenside
        if(ourCastlingMask&0b0101 && !(all&(0b111ull<<(kingSq-3))))
            *movePtr++=packMove_flag(kingSq,kingSq-2,FLAG_CASTLING);
    }
}

void generatePseudoLegalCaptures(Move*& movePtr,Position *pos,Bitboard evasionMask=FULL){

    Side side=pos->turn;

    Bitboard white=pos->colors[WHITE];
    Bitboard black=pos->colors[BLACK];

    Bitboard friendly=side?white:black;

    Bitboard enemy=side?black:white;
    Bitboard all=white|black;
    Bitboard empty=~all;

    Bitboard pawns=pos->pieces[PAWN]&friendly;

    //during check, gen all EP moves anyways
    Bitboard passant=pos->enpassant==-1?0:1ull<<pos->enpassant;
    Bitboard pawnCapEast=(side?(pawns&not_fileH)<<9:(pawns&not_fileH)>>7)&((enemy&evasionMask)|passant);
    Bitboard pawnCapWest=(side?(pawns&not_fileA)<<7:(pawns&not_fileA)>>9)&((enemy&evasionMask)|passant);

    Dir up=side?NORTH:SOUTH;
    Dir upwest=side?NORTHWEST:SOUTHWEST;
    Dir upeast=side?NORTHEAST:SOUTHEAST;

    while(pawnCapEast){
        Sq sq=poplsb(pawnCapEast);
        if(sq<8||sq>55){
            *movePtr++=packMove_promotion(sq-upeast,sq,3);
            *movePtr++=packMove_promotion(sq-upeast,sq,2);
            *movePtr++=packMove_promotion(sq-upeast,sq,1);
            *movePtr++=packMove_promotion(sq-upeast,sq,0);
        }else if(pos->enpassant==sq)
            *movePtr++=packMove_flag(sq-upeast,sq,FLAG_EN_PASSANT);
        else
            *movePtr++=packMove(sq-upeast,sq);
    }
    while(pawnCapWest){
        Sq sq=poplsb(pawnCapWest);
        if(sq<8||sq>55){
            *movePtr++=packMove_promotion(sq-upwest,sq,3);
            *movePtr++=packMove_promotion(sq-upwest,sq,2);
            *movePtr++=packMove_promotion(sq-upwest,sq,1);
            *movePtr++=packMove_promotion(sq-upwest,sq,0);
        }else if(pos->enpassant==sq)
            *movePtr++=packMove_flag(sq-upwest,sq,FLAG_EN_PASSANT);
        else
            *movePtr++=packMove(sq-upwest,sq);
    }

    Bitboard knights=pos->pieces[KNIGHT]&friendly;
    while(knights){
        Sq sq=poplsb(knights);
        Bitboard attacks=knightAttacks[sq]&enemy&evasionMask;

        while(attacks){
            Sq to=poplsb(attacks);
            *movePtr++=packMove(sq,to);
        }
    }

    Bitboard bishops=pos->pieces[BISHOP]&friendly;
    while(bishops){
        Sq sq=poplsb(bishops);
        Bitboard attacks=getBishopAttacks(sq,all)&enemy&evasionMask;
        
        while(attacks){
            Sq to=poplsb(attacks);
            *movePtr++=packMove(sq,to);
        }
    }

    Bitboard rooks=pos->pieces[ROOK]&friendly;
    while(rooks){
        Sq sq=poplsb(rooks);
        Bitboard attacks=getRookAttacks(sq,all)&enemy&evasionMask;
        
        while(attacks){
            Sq to=poplsb(attacks);
            *movePtr++=packMove(sq,to);
        }
    }

    Bitboard queens=pos->pieces[QUEEN]&friendly;
    while(queens){
        Sq sq=poplsb(queens);
        Bitboard attacks=(getBishopAttacks(sq,all)|getRookAttacks(sq,all))&enemy&evasionMask;
        
        while(attacks){
            Sq to=poplsb(attacks);
            *movePtr++=packMove(sq,to);
        }
    }

    Sq kingSq=counttrailingzeros(pos->pieces[KING]&friendly);
    Bitboard attacks=kingAttacks[kingSq]&enemy;

    while(attacks){
        Sq to=poplsb(attacks);
        *movePtr++=packMove(kingSq,to);
    }
}

void generatePseudoLegalKingMoves(Move*& movePtr,Position *pos,Sq kingSq){

    Bitboard not_friendly=~pos->colors[pos->turn];
    Bitboard attacks=kingAttacks[kingSq]&not_friendly;

    while(attacks){
        Sq to=poplsb(attacks);
        *movePtr++=packMove(kingSq,to);
    }
}
void generatePseudoLegalKingCaptures(Move*& movePtr,Position *pos,Sq kingSq){

    Bitboard enemy=pos->colors[pos->turn^1];
    Bitboard attacks=kingAttacks[kingSq]&enemy;

    while(attacks){
        Sq to=poplsb(attacks);
        *movePtr++=packMove(kingSq,to);
    }
}

Bitboard sqPossiblePinners(Position *pos,Sq sq,int side){
    
    //find enemy sliders that are capable of possibly pinning a piece if sq is the king sq. exclude contact checks
    Bitboard color=pos->colors[side];

    Bitboard bishops=pos->pieces[BISHOP]&color;
    Bitboard rooks=pos->pieces[ROOK]&color;
    Bitboard queens=pos->pieces[QUEEN]&color;
    //xray through our own pieces: the only blockers are the enemy pieces
    Bitboard blockers=color;

    return(
        getBishopAttacks(sq,blockers)&(bishops|queens)
        |
        getRookAttacks(sq,blockers)&(rooks|queens)
    )&(~kingAttacks[sq]);
}

bool isMoveLegal(Position *pos,Move move,Sq kingSq){
    //run for moves filtered: EP, king moves + castling, or pinned piece moves 

    Sq from=move&63,to=(move>>6)&63;
    int flag=move>>14;

    Side opp=pos->turn^1;

    //if move is a king move, make sure it doesnt move to an attacked sq
    if(from==kingSq){
        
        if(flag!=FLAG_CASTLING)
            //note: if the king is attacked by a ray and moves backwards away from the checker, but stays on that ray, the ghost king on the bitboard will block the check for the destination sq. clear the ghost king from the blockers BB during this detection.
            return !sqAttacked(pos,to,opp,1ull<<kingSq);
        else
            //castling: check if destination and that 1 square in the path isnt attacked. there shouldnt be edge cases with ghost kings like above(ghost kings blocking a horizontal ray here dont exist because it would be check). castling during check already filtered out by now(they are not generated during pseudo-legals gen)
            return !sqAttacked(pos,to,opp) && !sqAttacked(pos,(from+to)/2,opp);

    }else if(flag==FLAG_EN_PASSANT){
        
        //en passant: check if king not left attacked when the pawn is captured (remove it from the blockers) AND the capturing pawn is moved to the EP square. should basically just be specifically optimized make/check/undo pipeline. automatically handles pinned EP moves too: a legal pinned ep exists if the capturing pawn is on the pin ray, landing on the EP square which still blocks a diagonal check/pin
        Bitboard epSqBB=1ull<<(to+(pos->turn?-8:8));
        return !sqAttacked(pos,kingSq,opp,epSqBB|(1ull<<from)|(1ull<<to),~epSqBB);
    }
    
    Bitboard toBB=1ull<<to;

    //if move is not any of the above, its the move(non-EP) of a pinned piece. restrict the movement to the pin line (ray from the sq to the checker, extending beyond the checker(tho this doesnt matter))
    return rays[kingSq][from]&toBB;
}

void generateLegalMoves(Move*& movePtr,Position *pos,Bitboard checkers,int checkDegree){
    
    Move* ptr=movePtr;

    //start by computing pinned pieces:
    Bitboard friendly=pos->colors[pos->turn];
    Sq kingSq=counttrailingzeros(pos->pieces[KING]&friendly);
    Bitboard possiblePinners=sqPossiblePinners(pos,kingSq,pos->turn^1);
    Bitboard pinned=0;

    //if a possible pinner pins exactly one piece, that pinner is a true pinner and that pinned piece is pinned
    while(possiblePinners){
        Sq pinSq=poplsb(possiblePinners);
        Bitboard pinLine=betweens[kingSq][pinSq];
        Bitboard possiblePinned=pinLine&friendly;
        if(popcount(possiblePinned)==1)
            pinned|=possiblePinned;
    }

    //generate pseudo-legal moves to be filtered below:
    //no check: all psuedo-legals
    //single check: psuedo-legals for king move + blocking or capturing checking piece
    //double check: psuedo-legals for only king moves
    if(checkDegree==0)
        generatePseudoLegalMoves(movePtr,pos,FULL);
    else if(checkDegree==1){
        Sq checkerSq=counttrailingzeros(checkers);
        generatePseudoLegalMoves(movePtr,pos,betweens[kingSq][checkerSq]|(1ull<<checkerSq));
    }else
        generatePseudoLegalKingMoves(movePtr,pos,kingSq);

    while(ptr!=movePtr){
        Sq from=(*ptr)&63;
        int flag=(*ptr)>>14;
        Bitboard fromBB=1ull<<from;

        if((pinned&fromBB || from==kingSq || flag==FLAG_EN_PASSANT) && !isMoveLegal(pos,*ptr,kingSq))
            *ptr=*(--movePtr);
        else
            ptr++;
    }
}

void generateLegalCaptures(Move*& movePtr,Position *pos,Bitboard checkers,int checkDegree){
    //same as generateLegalMoves, but only generate captures via a mask
    Move* ptr=movePtr;

    Bitboard friendly=pos->colors[pos->turn];
    Bitboard enemy=pos->colors[pos->turn^1];
    Sq kingSq=counttrailingzeros(pos->pieces[KING]&friendly);
    Bitboard possiblePinners=sqPossiblePinners(pos,kingSq,pos->turn^1);
    Bitboard pinned=0;

    while(possiblePinners){
        Sq pinSq=poplsb(possiblePinners);
        Bitboard pinLine=betweens[kingSq][pinSq];
        Bitboard possiblePinned=pinLine&friendly;
        if(popcount(possiblePinned)==1)
            pinned|=possiblePinned;
    }

    if(checkDegree==0)
        generatePseudoLegalCaptures(movePtr,pos,FULL);
    else if(checkDegree==1){
        Sq checkerSq=counttrailingzeros(checkers);
        generatePseudoLegalCaptures(movePtr,pos,betweens[kingSq][checkerSq]|(1ull<<checkerSq));
    }else
        generatePseudoLegalKingCaptures(movePtr,pos,kingSq);

    while(ptr!=movePtr){
        Sq from=(*ptr)&63;
        int flag=(*ptr)>>14;
        Bitboard fromBB=1ull<<from;

        if((pinned&fromBB || from==kingSq || flag==FLAG_EN_PASSANT) && !isMoveLegal(pos,*ptr,kingSq))
            *ptr=*(--movePtr);
        else
            ptr++;
    }
}

void makeMove(UndoState*& undoPtr,Position *pos,Move move,Worker* worker){
    pos->ply++;

    Sq from=move&63,to=(move>>6)&63;
    int flag=move>>14;

    Piece pieceMoving=pos->board[from];
    int pieceMovingZobristIndex=pieceMoving-1;
    //type of piece moved and its color
    int pieceBB=(pieceMoving-1)%6,
        colorBB=pos->turn,
        oppColorBB=colorBB^1;

    Bitboard fromBB=1ull<<from;
    Bitboard toBB=1ull<<to;

    //before key is updated, store it
    undoPtr->key=pos->key;
    undoPtr->pawnKey=pos->pawnKey;
    undoPtr->nonPawnKey[0]=pos->nonPawnKey[0];
    undoPtr->nonPawnKey[1]=pos->nonPawnKey[1];
    undoPtr->minorKey=pos->minorKey;
    undoPtr->majorKey=pos->majorKey;
    
    //update board array, note capture
    Piece captured=pos->board[to];
    pos->board[to]=pieceMoving;
    pos->board[from]=0;

    Key zobPieceFrom=zobristPieceSq[pieceMovingZobristIndex][from];         
    Key zobPieceTo=zobristPieceSq[pieceMovingZobristIndex][to];
    pos->key^=zobPieceFrom;
    pos->key^=zobPieceTo;


    //update state info for undo(now that captured piece is defined)
    undoPtr->move=move;
    undoPtr->castling=pos->castling;
    undoPtr->enpassant=pos->enpassant;
    undoPtr->halfmoves=pos->halfmoves;
    undoPtr->moved=pieceMoving;
    undoPtr->captured=captured;
    undoPtr->continuationHistory=worker->continuationHistory[pieceMovingZobristIndex*64+to];
    
    #if USE_NNUE
        //accums are stored similarly
        undoPtr->whiteAccumulator=pos->whiteAccumulator;
        undoPtr->blackAccumulator=pos->blackAccumulator;
    #endif

    undoPtr++;
    
    #if USE_NNUE
        //now that the accums r saved, updating them starts here and in other cases, below
        addToNNUE(pos,pieceMovingZobristIndex,to);
        subFromNNUE(pos,pieceMovingZobristIndex,from);
    #endif

    //increment halfmove, reset if pawn moved
    pos->halfmoves++;

    if(pos->enpassant!=-1)
        pos->key^=zobristEnpassant[pos->enpassant&7];
    pos->enpassant=-1;

    if(pieceBB==PAWN){
        //first, update pawn key cuz a pawn moved    
        pos->pawnKey^=zobPieceFrom;
        pos->pawnKey^=zobPieceTo;

        pos->halfmoves=0;
        //update en passant square: detect double push. hash if en passant is truly available. doing it this way, ep is stored ONLY if its possible next move, and not whether it lables a target ep square (generally cleaner). but it saves a bit of computation when hashing (detecting possible ep when hashing + clearing the pe hash) as opposed to here (detecting possible ep only when double push happens)
        if((from^to)==16){
            
            Sq epSq=to+(pos->turn?-8:8);
            Bitboard sqMask=1ull<<epSq;
            Bitboard pawns=pos->pieces[PAWN]&pos->colors[oppColorBB];
            Bitboard capEast=oppColorBB?(sqMask&not_fileH)>>7:(sqMask&not_fileH)<<9;
            Bitboard capWest=oppColorBB?(sqMask&not_fileA)>>9:(sqMask&not_fileA)<<7;
            bool possibleEP=(capEast|capWest)&pawns;

            if(possibleEP){
                pos->enpassant=epSq;
                pos->key^=zobristEnpassant[to&7];
            }
        }
    }else{
        pos->nonPawnKey[colorBB]^=zobPieceFrom;
        pos->nonPawnKey[colorBB]^=zobPieceTo;
        //not a pawn move, update those other corrhist keys
        if(pieceBB<=BISHOP){
            pos->minorKey^=zobPieceFrom;
            pos->minorKey^=zobPieceTo;
        }else{
            pos->majorKey^=zobPieceFrom;
            pos->majorKey^=zobPieceTo;
        }
    }

    //if capturing
    if(captured){

        pos->halfmoves=0;
        //handle bitboards after ordinary capture: remove captured piece
        int capturedBB=(captured-1)%6;
        pos->pieces[capturedBB]^=toBB;
        pos->colors[oppColorBB]^=toBB;
        
        Key zobPieceCaptured=zobristPieceSq[captured-1][to];
        pos->key^=zobPieceCaptured;

        //update the other silly keys
        if(capturedBB==PAWN)
            pos->pawnKey^=zobPieceCaptured;
        else{
            pos->nonPawnKey[oppColorBB]^=zobPieceCaptured;
            if(capturedBB<=BISHOP)
                pos->minorKey^=zobPieceCaptured;
            else
                pos->majorKey^=zobPieceCaptured;
        }

        #if USE_NNUE
            subFromNNUE(pos,captured-1,to);
        #endif

        //if didnt capture, handle castling (they're mutually exclusive)
    }else if(flag==FLAG_CASTLING){
        //here, only the rook needs to be moved accordingly. the king's move is handled in this func alr
        bool castledKingside=to>from;
        Sq rookSq=castledKingside?from+3:from-4;
        Sq rookTo=castledKingside?from+1:from-1;
        
        Bitboard rookSqBB=1ull<<rookSq;
        Bitboard rookToBB=1ull<<rookTo;

        Piece rookPiece=pos->board[rookSq];
        int rookZobristIndex=rookPiece-1;

        pos->board[rookTo]=rookPiece;
        pos->board[rookSq]=0;
        pos->pieces[ROOK]^=rookSqBB|rookToBB;
        //smart: use colorBB because the rook's color is the same as the king
        pos->colors[colorBB]^=rookSqBB|rookToBB;

        Key zobRookFrom=zobristPieceSq[rookZobristIndex][rookSq];
        Key zobRookTo=zobristPieceSq[rookZobristIndex][rookTo];
        pos->key^=zobRookFrom;
        pos->key^=zobRookTo;

        pos->nonPawnKey[colorBB]^=zobRookFrom;
        pos->nonPawnKey[colorBB]^=zobRookTo;
        pos->majorKey^=zobRookFrom;
        pos->majorKey^=zobRookTo;
        
        #if USE_NNUE
            addToNNUE(pos,rookZobristIndex,rookTo);
            subFromNNUE(pos,rookZobristIndex,rookSq);
        #endif

    //EP doesn't count as a capture above because the above looked at the destination sq for an existing piece
    }else if(flag==FLAG_EN_PASSANT){
        //en passant: remove that pawn
        Sq sqOfPassantedPawn=to+(pos->turn?-8:8);
        Bitboard passBB=1ull<<sqOfPassantedPawn;
        //bitboards here should be right: always PAWN and opposite color of the piece moved(the pawn thats enpassant-ing) 
        pos->board[sqOfPassantedPawn]=0;
        pos->pieces[PAWN]^=passBB;
        pos->colors[oppColorBB]^=passBB;

        //pos->turn?6:0
        //update pawn key too, since ep always captures a pawn
        Key zobPiece=zobristPieceSq[pos->turn*6][sqOfPassantedPawn];
        pos->key^=zobPiece;
        pos->pawnKey^=zobPiece;
        
        #if USE_NNUE
            subFromNNUE(pos,pos->turn*6,sqOfPassantedPawn);
        #endif
    }

    //update bitboards for move accordingly
    pos->pieces[pieceBB]^=fromBB|toBB;
    pos->colors[colorBB]^=fromBB|toBB;

    if(flag==FLAG_PROMOTION){
        int prom=(move>>12)&3;
        //remove the pawn
        pos->pieces[pieceBB]^=toBB;
        pos->key^=zobPieceTo;
        pos->pawnKey^=zobPieceTo;

        //add the new piece
        Piece newPiece=prom+(pos->turn?KNIGHT_W:KNIGHT_B);
        pos->board[to]=newPiece;
        pos->pieces[prom+1]|=toBB;

        Key zobNewPiece=zobristPieceSq[newPiece-1][to];
        pos->key^=zobNewPiece;
        prom+=KNIGHT;
        pos->nonPawnKey[colorBB]^=zobNewPiece;
        if(prom<=BISHOP)
            pos->minorKey^=zobNewPiece;
        else
            pos->majorKey^=zobNewPiece;
        
        #if USE_NNUE
            subFromNNUE(pos,pieceMovingZobristIndex,to);
            addToNNUE(pos,newPiece-1,to);
        #endif
    }

    #if USE_NNUE
        //do horizontal mirroring: if the king move crosses the border then refresh that players accumulator
        if(pieceMoving==KING_W && ((to&7)>=4)!=pos->mirrorWhite)
            refreshWhiteAccumulator(pos);
        if(pieceMoving==KING_B && ((to&7)>=4)!=pos->mirrorBlack)
            refreshBlackAccumulator(pos);
    #endif

    //update castling rights
    pos->key^=zobristCastling[pos->castling];
    pos->castling&=castlingRights[from]&castlingRights[to];
    pos->key^=zobristCastling[pos->castling];
    
    //unupdated turn is used above. now update for next turn
    pos->turn^=1;
    pos->key^=zobristSide;
}

void undoMove(UndoState*& undoPtr,Position *pos){
    pos->ply--;

    UndoState *state=--undoPtr;
    
    Move move=state->move;
    Sq from=move&63,to=(move>>6)&63;
    int flag=move>>14;

    Bitboard fromBB=1ull<<from;
    Bitboard toBB=1ull<<to;

    //restore the obvious or stored things
    pos->turn^=1;
    pos->castling=state->castling;
    pos->enpassant=state->enpassant;
    pos->halfmoves=state->halfmoves;
    pos->key=state->key;
    pos->pawnKey=state->pawnKey;
    pos->nonPawnKey[0]=state->nonPawnKey[0];
    pos->nonPawnKey[1]=state->nonPawnKey[1];
    pos->minorKey=state->minorKey;
    pos->majorKey=state->majorKey;

    #if USE_NNUE
        pos->whiteAccumulator=state->whiteAccumulator;
        pos->blackAccumulator=state->blackAccumulator;
    #endif

    Piece pieceMoving=state->moved;

    //undo the move (remove from "to", add to "from"). careful of promotion, as pos->board[to] is incorrectly the promoted piece
    int pieceBB=(pieceMoving-1)%6,
        colorBB=pos->turn,
        oppColorBB=colorBB^1;

    pos->board[from]=pieceMoving;
    pos->board[to]=state->captured;

    if(flag!=FLAG_PROMOTION){
        //if not promotion, remove from "to", add to "from" as normal
        pos->pieces[pieceBB]^=fromBB|toBB;
    }else{
        //if promotion, remove whatever piece was promoted, add back the pawn
        pos->pieces[((move>>12)&3)+1]^=toBB;
        pos->pieces[PAWN]|=fromBB;
        pos->board[from]=pos->turn?PAWN_W:PAWN_B;
    }
    pos->colors[colorBB]^=fromBB|toBB;
    
    //if it was a capture, add back the captured piece. pos->board[to] alr done above
    if(state->captured){
        pos->pieces[(state->captured-1)%6]|=toBB;
        pos->colors[oppColorBB]|=toBB;
    }else if(flag==FLAG_CASTLING){
        //castling: the king alr done above. move the rook back
        bool castledKingside=to>from;
        Sq rookSq=castledKingside?from+1:from-1;
        Sq rookHome=castledKingside?from+3:from-4;

        Bitboard rookBBflip=(1ull<<rookSq)|(1ull<<rookHome);

        pos->pieces[ROOK]^=rookBBflip;
        pos->colors[colorBB]^=rookBBflip;

        pos->board[rookHome]=pos->turn?ROOK_W:ROOK_B;
        pos->board[rookSq]=0;
    }else if(flag==FLAG_EN_PASSANT){
        //en passant: add back that pawn
        Sq sqOfPassantedPawn=to+(pos->turn?-8:8);
        Bitboard passBB=1ull<<sqOfPassantedPawn;
        pos->pieces[PAWN]|=passBB;
        pos->colors[oppColorBB]|=passBB;
        pos->board[sqOfPassantedPawn]=pos->turn?PAWN_B:PAWN_W;
    }

    #if USE_NNUE
        //make sure mirror booleans are updated back: re-derive them
        if(pieceMoving==KING_W)
            pos->mirrorWhite=(from&7)>=4;
        if(pieceMoving==KING_B)
            pos->mirrorBlack=(from&7)>=4;
    #endif
}

void makeNullMove(UndoState*& undoPtr,Position *pos){
    pos->ply++;

    undoPtr->key=pos->key;
    //null moves don't change these keys, so this may not be necessary. but it's here anyway
    undoPtr->pawnKey=pos->pawnKey;
    undoPtr->nonPawnKey[0]=pos->nonPawnKey[0];
    undoPtr->nonPawnKey[1]=pos->nonPawnKey[1];
    undoPtr->minorKey=pos->minorKey;
    undoPtr->majorKey=pos->majorKey;
    
    undoPtr->move=0;
    undoPtr->castling=pos->castling;
    undoPtr->enpassant=pos->enpassant;
    undoPtr->halfmoves=pos->halfmoves;
    undoPtr->captured=0;
    
    #if USE_NNUE
        undoPtr->whiteAccumulator=pos->whiteAccumulator;
        undoPtr->blackAccumulator=pos->blackAccumulator;
    #endif

    undoPtr++;

    if(pos->enpassant!=-1)
        pos->key^=zobristEnpassant[pos->enpassant&7];
    pos->enpassant=-1;

    pos->turn^=1;
    pos->key^=zobristSide;
}

void undoNullMove(UndoState*& undoPtr,Position *pos){
    pos->ply--;
    undoPtr--;
    pos->turn^=1;
    pos->key=undoPtr->key;
    pos->pawnKey=undoPtr->pawnKey;
    pos->nonPawnKey[0]=undoPtr->nonPawnKey[0];
    pos->nonPawnKey[1]=undoPtr->nonPawnKey[1];
    pos->minorKey=undoPtr->minorKey;
    pos->majorKey=undoPtr->majorKey;
    pos->enpassant=undoPtr->enpassant;
    
    #if USE_NNUE
    pos->whiteAccumulator=undoPtr->whiteAccumulator;
    pos->blackAccumulator=undoPtr->blackAccumulator;
    #endif
}

//to be called after makeMove for an "official" board state
void updatePastPositionRepetition(Position *pos){
    if(pos->halfmoves==0){
        pos->pastRepetitionLength=1;
        pos->pastRepetitionKeys[0]=pos->key;
    }else{
        //shift the array right
        for(int i=pos->pastRepetitionLength;i>0;i--)
            pos->pastRepetitionKeys[i]=pos->pastRepetitionKeys[i-1];

        //the root is updated:
        pos->pastRepetitionKeys[0]=pos->key;
        pos->pastRepetitionLength++;
    }
}

void resetPastPositionRepetition(Position *pos){
    pos->pastRepetitionLength=1;
    pos->pastRepetitionKeys[0]=pos->key;
}

//50 move rule or threefold repetition
bool isDrawByRule(UndoState*& undoPtr,Position *pos){
    
    //repetition physically requires at least 8 reversible moves
    if(pos->halfmoves<8)return false;
    
    //50 move rule
    if(pos->halfmoves>=100)return true;

    //look through the game history: undoStack has the keys explored during the search, pos->pastRepetitionKeys has the relevant keys of the game moves
    // root=pos->pastRepetitionKeys[0]=undoStack[0]
    // pos->ply=undoStack's length

    Key key=pos->key;
    int repCount=0;
    int howManyMovesBack=pos->halfmoves;

    int undoStackLimit=min(howManyMovesBack,pos->ply);
    for(int i=2;i<undoStackLimit;i+=2)
        if((undoPtr-i)->key==key)
            if(++repCount>=2)return true;

    //if we call this function at root, the root node will be counted cuz the root is included below. if so, ++repCount has to be >=3. repCount-- does the trick
    if(pos->ply==0)repCount--;

    int gameHistoryLimit=min((howManyMovesBack-undoStackLimit)+1,pos->pastRepetitionLength);
    for(int i=undoStackLimit&1;i<gameHistoryLimit;i+=2)
        if(pos->pastRepetitionKeys[i]==key)
            if(++repCount>=2)return true;

    return false;
}


std::string uciMove(Position *pos,Move move){

    std::string out="";

    Sq from=move&63,to=(move>>6)&63;
    int flag=move>>14;

    out+='a'+(from&7);
    out+=(from>>3)+'1';
    out+='a'+(to&7);
    out+=(to>>3)+'1';

    if(flag==FLAG_PROMOTION)
        switch((move>>12)&3){
            case 0:out+='n';break;
            case 1:out+='b';break;
            case 2:out+='r';break;
            case 3:out+='q';break;
        }
    
    return out;
}

std::string algMove(Move *& movePtr,UndoState*& undoPtr,Position *pos,Move move,Worker* worker){
    //converts a move into algebraic notation. fully compliant with SAN. sigh why

    std::string out;

    Sq from=move&63,to=(move>>6)&63;
    int flag=move>>14;
    Piece pieceMoving=pos->board[from];
    Piece pieceType=(pieceMoving-1)%6;

    if(flag==FLAG_CASTLING)
        //cant return here yet: edge case of castle check + checkmate
        out=to>from?"O-O":"O-O-O";
    else{

        switch(pieceType){
            case KNIGHT:out+="N";break;
            case BISHOP:out+="B";break;
            case ROOK:out+="R";break;
            case QUEEN:out+="Q";break;
            case KING:out+="K";break;
        }

        //disambiguation
        if(pieceType!=PAWN && pieceType!=KING){
            
            bool conflict=false,disamFile=false,disamRank=false;

            Sq kingSq=counttrailingzeros(pos->pieces[KING]&pos->colors[pos->turn]);
            Bitboard checkers=sqAttackers(pos,kingSq,pos->turn^1);
            Move* startMoveList=movePtr;
            generateLegalMoves(movePtr,pos,checkers,popcount(checkers));
            Move* endMoveList=movePtr;
            for(Move* m=startMoveList;m<endMoveList;m++){
                Sq mFrom=*m&63,mTo=(*m>>6)&63;
                //if same piece type can go to the same square, test for ambuiguity and mark
                if(*m!=move && pos->board[mFrom]==pieceMoving && mTo==to){
                    conflict=true;
                    if((mFrom>>3)==(from>>3))disamFile=true;
                    if((mFrom&7)==(from&7))disamRank=true;
                }
            }
            movePtr=startMoveList;
            
            //if there is a conflict, there MUST be some sort of disambiguation needed
            if(conflict){
                //if theres a conflict but no matches on either axis, its the case of pieces not aligned with each other on any axis: unaligned knights or diagonal lines. only file disambiguation needed, and it takes precedence
                if(!disamFile && !disamRank){
                    out+='a'+(from&7);
                }else{
                    //otherwise, disambiguate per axis
                    if(disamFile)out+='a'+(from&7);
                    if(disamRank)out+=(from>>3)+'1';
                }
            }
        }

        if(pos->board[to]||flag==FLAG_EN_PASSANT){
            if(pieceType==PAWN)
                out+='a'+(from&7);
            out+="x";
        }
        out+='a'+(to&7);
        out+=(to>>3)+'1';

        if(flag==FLAG_PROMOTION)
            switch((move>>12)&3){
                case 0:out+="=N";break;
                case 1:out+="=B";break;
                case 2:out+="=R";break;
                case 3:out+="=Q";break;
            }
    }
    
    //check for check and checkmate: make the move, determine if opp is in check, if so, movegen to see if there are legals
    makeMove(undoPtr,pos,move,worker);
    Sq oppKingSq=counttrailingzeros(pos->pieces[KING]&pos->colors[pos->turn]);
    Bitboard oppCheckers=sqAttackers(pos,oppKingSq,pos->turn^1);
    if(oppCheckers){
        Move* startMoveList=movePtr;
        generateLegalMoves(movePtr,pos,oppCheckers,popcount(oppCheckers));
        if(movePtr==startMoveList)
            out+="#";
        else    
            out+="+";
        movePtr=startMoveList;
    }
    undoMove(undoPtr,pos);

    return out;
}

std::string relaxedAlg(std::string alg){
    
    std::string out="";
    for(size_t i=0;i<alg.size();i++){

        char c=alg[i];
        
        if(c>='A'&&c<='Z')c+='a'-'A';

        if(c=='0')c='o';

        if((c>='a'&&c<='h')||(c>='1'&&c<='8')||(c=='n'||c=='b'||c=='r'||c=='q'||c=='k'||c=='o'))
            out+=c;
    }
    return out;
}

void printMove(Move*& movePtr,UndoState*& undoPtr,Position *pos,Move move,Worker* worker,int count=0,uint64_t nodes=0){

    std::string alg=algMove(movePtr,undoPtr,pos,move,worker);

    if(count>0)
        std::cout << count <<". ";
    std::cout << uciMove(pos,move) << " " << alg << " " << relaxedAlg(alg);
    if(nodes>0ull)
        std::cout << "  " << nodes;
    std::cout << "\n";
}

Move stringToMove(Position *pos,std::string move){
    //turns a uci move string into a packed move, not necessarily legal, pseudo-legal, nor correct

    Sq from=(move[0]-'a')+(move[1]-'1')*8,
        to=(move[2]-'a')+(move[3]-'1')*8;
    
    if(from<0||to<0||from>63||to>63)return 0;

    Piece pieceType=(pos->board[from]-1)%6;

    //detect promotion
    if(move[4]=='n')return packMove_promotion(from,to,0);
    else if(move[4]=='b')return packMove_promotion(from,to,1);
    else if(move[4]=='r')return packMove_promotion(from,to,2);
    else if(move[4]=='q')return packMove_promotion(from,to,3);

    //detect EP: pawn captures empty square
    if(pieceType==PAWN && ((from&7)!=(to&7)) && pos->board[to]==0)return packMove_flag(from,to,FLAG_EN_PASSANT);

    //detect castling: king moves 2 squares
    if(pieceType==KING && (from-to==2||from-to==-2))return packMove_flag(from,to,FLAG_CASTLING);

    return packMove(from,to);
}

void setPositionKey(Position *pos){
    pos->key=0;
    pos->pawnKey=0;
    pos->nonPawnKey[0]=0;
    pos->nonPawnKey[1]=0;
    pos->minorKey=0;
    pos->majorKey=0;

    for(int i=0;i<64;i++){
        Piece p=pos->board[i];
        if(p){
            Key zobPiece=zobristPieceSq[p-1][i];
            pos->key^=zobPiece;
            Piece piece=(p-1)%6;
            if(piece==PAWN){
                pos->pawnKey^=zobPiece;
            }else{
                pos->nonPawnKey[p>=7?BLACK:WHITE]^=zobPiece;

                if(piece<=BISHOP)
                    pos->minorKey^=zobPiece;
                else
                    pos->majorKey^=zobPiece;
            }
        }
    }

    if(pos->enpassant>=0)
        pos->key^=zobristEnpassant[pos->enpassant&7];

    pos->key^=zobristCastling[pos->castling];

    if(pos->turn)
        pos->key^=zobristSide;
}

void initPosition(Position *pos){
    pos->turn=1;
    pos->pieces[PAWN]=0x00FF00000000FF00ull;
    pos->pieces[KNIGHT]=(1ull<<1)|(1ull<<6)|(1ull<<57)|(1ull<<62);
    pos->pieces[BISHOP]=(1ull<<2)|(1ull<<5)|(1ull<<58)|(1ull<<61);
    pos->pieces[ROOK]=(1ull<<0)|(1ull<<7)|(1ull<<56)|(1ull<<63);
    pos->pieces[QUEEN]=(1ull<<3)|(1ull<<59);
    pos->pieces[KING]=(1ull<<4)|(1ull<<60);
    pos->colors[WHITE]=0x000000000000ffffull;
    pos->colors[BLACK]=0xffff000000000000ull;
    pos->castling=15;
    pos->enpassant=-1;
    pos->halfmoves=0;

    for(int i=0;i<64;i++)
        pos->board[i]=0;
    for(int i=0;i<8;i++){
        pos->board[i+8]=PAWN_W;
        pos->board[i+48]=PAWN_B;
    }
    pos->board[1]=KNIGHT_W;
    pos->board[6]=KNIGHT_W;
    pos->board[57]=KNIGHT_B;
    pos->board[62]=KNIGHT_B;
    pos->board[2]=BISHOP_W;
    pos->board[5]=BISHOP_W;
    pos->board[58]=BISHOP_B;
    pos->board[61]=BISHOP_B;
    pos->board[0]=ROOK_W;
    pos->board[7]=ROOK_W;
    pos->board[56]=ROOK_B;
    pos->board[63]=ROOK_B;
    pos->board[3]=QUEEN_W;
    pos->board[59]=QUEEN_B;
    pos->board[4]=KING_W;
    pos->board[60]=KING_B;
    
    setPositionKey(pos);
    resetPastPositionRepetition(pos);
}

void initFen(Position *pos,const char* fen){
    pos->pieces[PAWN]=0;
    pos->pieces[KNIGHT]=0;
    pos->pieces[BISHOP]=0;
    pos->pieces[ROOK]=0;
    pos->pieces[QUEEN]=0;
    pos->pieces[KING]=0;
    pos->colors[WHITE]=0;
    pos->colors[BLACK]=0;

    for(int i=0;i<64;i++)
        pos->board[i]=0;

    int rank=7,file=0;
    while(*fen!=' '){

        if(isdigit(*fen)){
            file+=*fen-'0';
            fen++;
            continue;
        }else if(*fen=='/'){
            rank--;
            file=0;
            fen++;
            continue;
        }

        Sq sq=(rank<<3)+file;
        file++;

        char piece=tolower(*fen);
        if(piece=='p'){
            pos->pieces[PAWN]|=1ull<<sq;
            pos->board[sq]=PAWN_W;
        }else if(piece=='n'){
            pos->pieces[KNIGHT]|=1ull<<sq;
            pos->board[sq]=KNIGHT_W;
        }else if(piece=='b'){
            pos->pieces[BISHOP]|=1ull<<sq;
            pos->board[sq]=BISHOP_W;
        }else if(piece=='r'){
            pos->pieces[ROOK]|=1ull<<sq;
            pos->board[sq]=ROOK_W;
        }else if(piece=='q'){
            pos->pieces[QUEEN]|=1ull<<sq;
            pos->board[sq]=QUEEN_W;
        }else if(piece=='k'){
            pos->pieces[KING]|=1ull<<sq;
            pos->board[sq]=KING_W;
        }

        if(isupper(*fen))
            pos->colors[WHITE]|=1ull<<sq;
        else if(islower(*fen)){
            pos->colors[BLACK]|=1ull<<sq;
            pos->board[sq]+=PAWN_B-PAWN_W;
        }

        fen++;
    }

    fen++;
    pos->turn=(*fen=='w')?1:0;

    fen+=2;
    pos->castling=0;
    while(*fen!=' '){
        if(*fen=='K')pos->castling|=0b1000;
        if(*fen=='Q')pos->castling|=0b0100;
        if(*fen=='k')pos->castling|=0b0010;
        if(*fen=='q')pos->castling|=0b0001;
        fen++;
    }

    fen++;
    pos->enpassant=-1;
    if(*fen!='-'){
        pos->enpassant=*fen-'a';
        fen++;
        pos->enpassant+=(*fen-'1')*8;

        Sq epSq=pos->enpassant;
        Bitboard sqMask=1ull<<epSq;
        Side opp=pos->turn^1;
        Bitboard pawns=pos->pieces[PAWN]&pos->colors[opp];
        Bitboard capEast=opp?(sqMask&not_fileH)>>7:(sqMask&not_fileH)<<9;
        Bitboard capWest=opp?(sqMask&not_fileA)>>9:(sqMask&not_fileA)<<7;
        bool possibleEP=(capEast|capWest)&pawns;
        if(!possibleEP)
            pos->enpassant=-1;
    }

    fen+=2;
    pos->halfmoves=0;
    while(isdigit(*fen)){
        pos->halfmoves=pos->halfmoves*10+*fen-'0';
        fen++;
    }

    setPositionKey(pos);
    resetPastPositionRepetition(pos);
}

void clearTT(){
    memset(TT,0,sizeof(TT));
    memset(TT_perft,0,sizeof(TT_perft));
}

void Worker::initSearch(Position *pos){

    ttAge++;

    movePtr=moveList;
    undoPtr=undoStack;
    nodeCount=0;
    out_nodes.store(nodeCount,std::memory_order_relaxed);
    pos->ply=0;

    memset(killerMoves,0,sizeof(killerMoves));
    memset(countermoveHistory,0,sizeof(countermoveHistory));

    for(int i=0;i<12*64;i++){
        historyTable[i]/=2;
        //inject some noise for threads
        historyTable[i]+=(id>0)*(((id*147)^(i*9))&3);

        for(int j=0;j<6;j++)
            captureHistoryTable[i][j]/=2;
            
        for(int j=0;j<12*64;j++)
            continuationHistory[i][j]/=2;
    }

    #if USE_NNUE
        resetNNUE(pos);
    #endif
}

void Worker::clearHistories(){
    
    memset(historyTable,0,sizeof(historyTable));
    memset(captureHistoryTable,0,sizeof(captureHistoryTable));
    memset(continuationHistory,0,sizeof(continuationHistory));
    memset(killerMoves,0,sizeof(killerMoves));
    memset(countermoveHistory,0,sizeof(countermoveHistory));
    memset(pawnCorrectionHistory,0,sizeof(pawnCorrectionHistory));
    memset(nonpawnCorrectionHistory,0,sizeof(nonpawnCorrectionHistory));
    memset(minorCorrectionHistory,0,sizeof(minorCorrectionHistory));
    memset(majorCorrectionHistory,0,sizeof(majorCorrectionHistory));
}

void assert(bool b,Position *pos,Key key=0){
    if(!b){
        printf("ASSERT FAILED\n\n");
        printPosition(pos);

        if(key!=0)
            std::cout << uint64ToHex(key) << "\n";

        //print the move path from undostack
        // for(int i=0;i<undoPtr-undoStack;i++){
        //     Move move=undoStack[i].move;
        //     std::cout << uciMove(pos,move) << " ";
        // }
        printf("\n");

        exit(1);
    }
}

void verifyPosition(Position *pos){

    Key key=pos->key;
    Key pawnKey=pos->pawnKey;
    Key nonPawnKey0=pos->nonPawnKey[0];
    Key nonPawnKey1=pos->nonPawnKey[1];
    Key minorKey=pos->minorKey;
    Key majorKey=pos->majorKey;

    setPositionKey(pos);
    assert(pos->key==key && pos->pawnKey==pawnKey && pos->nonPawnKey[0]==nonPawnKey0 && pos->nonPawnKey[1]==nonPawnKey1 && pos->minorKey==minorKey && pos->majorKey==majorKey,pos,key);

    Bitboard all=pos->colors[WHITE]|pos->colors[BLACK];
    Bitboard piecesBB=0;
    for(int i=0;i<6;i++)
        piecesBB|=pos->pieces[i];
    assert(all==piecesBB,pos);

    for(int i=0;i<64;i++){
        Piece p=pos->board[i];
        if(p!=0){
            assert((pos->pieces[BB_indexPiece[p]]&(1ull<<i))!=0,pos);
            assert((pos->colors[BB_indexColor[p]]&(1ull<<i))!=0,pos);
        }else{
            assert((pos->pieces[BB_indexPiece[p]]&(1ull<<i))==0,pos);
            assert((pos->colors[BB_indexColor[p]]&(1ull<<i))==0,pos);
        }
    }

    
    #if USE_NNUE
    Position temp=*pos;
    resetNNUE(&temp);

    for(int i=0;i<HIDDEN_SIZE;i++){
        assert(temp.whiteAccumulator.vals[i]==pos->whiteAccumulator.vals[i],pos);
        assert(temp.blackAccumulator.vals[i]==pos->blackAccumulator.vals[i],pos);
    }
    #endif
}

uint64_t Worker::perft(Position *pos,int depth,int startDepth){

    // if(depth<=0)return 1;

    verifyPosition(pos);

    //probe tt
    Key TTKey=pos->key&TTPerftSizeAnd;
    TTEntry_perft& entry=TT_perft[TTKey];
    if(pos->key==entry.key && entry.depth==depth)
        return entry.count;

    //bulk count
    if(depth==1){
        Sq kingSq=counttrailingzeros(pos->pieces[KING]&pos->colors[pos->turn]);
        Bitboard checkers=sqAttackers(pos,kingSq,pos->turn^1);
        int checkDegree=popcount(checkers);
        Move* startMoveList=movePtr;
        generateLegalMoves(movePtr,pos,checkers,checkDegree);
        return movePtr-startMoveList;
    }

    Sq kingSq=counttrailingzeros(pos->pieces[KING]&pos->colors[pos->turn]);
    Bitboard checkers=sqAttackers(pos,kingSq,pos->turn^1);
    int checkDegree=popcount(checkers);

    uint64_t count=0;

    Move* startMoveList=movePtr;
    generateLegalMoves(movePtr,pos,checkers,checkDegree);
    Move* endMoveList=movePtr;
    for(Move* move=startMoveList;move<endMoveList;move++){

        makeMove(undoPtr,pos,*move,this);

        uint64_t preCount=count;
        count+=perft(pos,depth-1,startDepth);
        undoMove(undoPtr,pos);

        //perft divide
        if(depth==startDepth)
            printMove(movePtr,undoPtr,pos,*move,this,move-startMoveList+1,count-preCount);
    }
    movePtr=startMoveList;

    entry.key=pos->key;
    entry.depth=depth;
    entry.count=count;

    return count;
}

constexpr Value toMoveBonus=20;
constexpr Score bishopPairBonus=S(17,25);
constexpr Score doubledPawnsPenalty=S(-20,-50);
constexpr Score passedPawnRankBonus[8]={S(0,0),S(10,28),S(17,33),S(25,41),S(45,60),S(90,115),S(175,205),S(0,0)};
constexpr Score isolatedPawnPenalty=S(-6,-16);
constexpr Score connectedPawnRankBonus[8]={S(0,0),S(2,3),S(3,5),S(6,7),S(8,9),S(13,20),S(40,50),S(0,0)};
constexpr Score backwardsPawnPenalty=S(-6,-15);
constexpr Score semiOpenFileRook=S(19,7);
constexpr Score openFileRook=S(48,29);
constexpr Score badBishopPerPawnPenalty=S(-3,-7);
constexpr Score knightOutpostBonus=S(32,20);
constexpr Score bishopOutpostBonus=S(25,16);
constexpr Score undefendedMinorPenalty=S(-15,-9);
constexpr int pawnShield1Safety=30;
constexpr int pawnShield2Safety=22;
constexpr int kingExposedFileSafety=-16;
constexpr int kingDiagonalSpaceSafety=-4;
constexpr int kingPawnTropisms[8]={0,18,15,12,9,6,3,0};
constexpr int kingPieceTropisms[8]={0,15,13,11,8,5,3,0};
constexpr int knightTropism=2;
constexpr int bishopTropism=2;
constexpr int rookTropism=3;
constexpr int queenTropism=4;
constexpr int minorAttackUnits=2;
constexpr int rookAttackUnits=3;
constexpr int queenAttackUnits=5;
constexpr int pawnThreatBonusMG=30;
constexpr int pawnThreatBonusEG=24;

constexpr Score MobilityBonus[][32]={//https://github.com/Stockfish-Classic/Stockfish-Classic/blob/33b1f98c556c68497b7a05327256d90f8f2bcc6a/src/evaluate.cpp but dampened 0.5x
    { S(0.5*-62,-81*0.5), S(0.5*-53,-56*0.5), S(0.5*-12,-31*0.5), S(0.5* -4,-16*0.5), S(0.5*  3,  5*0.5), S(0.5* 13, 11*0.5), // Knight
        S(0.5* 22, 17*0.5), S(0.5* 28, 20*0.5), S(0.5* 33, 25*0.5) },
    { S(0.5*-48,-59*0.5), S(0.5*-20,-23*0.5), S(0.5* 16, -3*0.5), S(0.5* 26, 13*0.5), S(0.5* 38, 24*0.5), S(0.5* 51, 42*0.5), // Bishop
        S(0.5* 55, 54*0.5), S(0.5* 63, 57*0.5), S(0.5* 63, 65*0.5), S(0.5* 68, 73*0.5), S(0.5* 81, 78*0.5), S(0.5* 81, 86*0.5),
        S(0.5* 91, 88*0.5), S(0.5* 98, 97*0.5) },
    { S(0.5*-60,-78*0.5), S(0.5*-20,-17*0.5), S(0.5*  2, 23*0.5), S(0.5*  3, 39*0.5), S(0.5*  3, 70*0.5), S(0.5* 11, 99*0.5), // Rook
        S(0.5* 22,103*0.5), S(0.5* 31,121*0.5), S(0.5* 40,134*0.5), S(0.5* 40,139*0.5), S(0.5* 41,158*0.5), S(0.5* 48,164*0.5),
        S(0.5* 57,168*0.5), S(0.5* 57,169*0.5), S(0.5* 62,172*0.5) },
    { S(0.5*-30,-48*0.5), S(0.5*-12,-30*0.5), S(0.5* -8, -7*0.5), S(0.5* -9, 19*0.5), S(0.5* 20, 40*0.5), S(0.5* 23, 55*0.5), // Queen
        S(0.5* 23, 59*0.5), S(0.5* 35, 75*0.5), S(0.5* 38, 78*0.5), S(0.5* 53, 96*0.5), S(0.5* 64, 96*0.5), S(0.5* 65,100*0.5),
        S(0.5* 65,121*0.5), S(0.5* 66,127*0.5), S(0.5* 67,131*0.5), S(0.5* 67,133*0.5), S(0.5* 72,136*0.5), S(0.5* 72,141*0.5),
        S(0.5* 77,147*0.5), S(0.5* 79,150*0.5), S(0.5* 93,151*0.5), S(0.5*108,168*0.5), S(0.5*108,168*0.5), S(0.5*108,171*0.5),
        S(0.5*110,182*0.5), S(0.5*114,182*0.5), S(0.5*114,192*0.5), S(0.5*116,219*0.5) }
};
constexpr int SafetyTable[100]={
    0,  0,   1,   2,   3,   5,   7,   9,  12,  15,
  18,  22,  26,  30,  35,  39,  44,  50,  56,  62,
  68,  75,  82,  85,  89,  97, 105, 113, 122, 131,
 140, 150, 169, 180, 191, 202, 213, 225, 237, 248,
 260, 272, 283, 295, 307, 319, 330, 342, 354, 366,
 377, 389, 401, 412, 424, 436, 448, 459, 471, 483,
 494, 500, 500, 500, 500, 500, 500, 500, 500, 500,
 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
 500, 500, 500, 500, 500, 500, 500, 500, 500, 500
};

struct TraceEval{
    uint32_t phase;
    Score pst[2];
    Score mobility[2];
    Score pawns[2];
    Score outpost[2];
    Score bishops[2];
    Score rookFiles[2];
    Score kingSafety[2];
    Value sum[2];
    Value beforeAdjust;
};
TraceEval trace;

uint32_t gamePhase(Position *pos){
    
    return 128-(((
        popcount(pos->pieces[KNIGHT])+
        popcount(pos->pieces[BISHOP])+
        popcount(pos->pieces[ROOK])*2+
        popcount(pos->pieces[QUEEN])*4
    )<<7)/24);
}

Score evaluateSide(Position *pos,Side side){
    Score score=0;

    Side opp=side^1;
    Bitboard friendly=pos->colors[side];
    Bitboard enemy=pos->colors[opp];
    Bitboard all=friendly|enemy;
    Bitboard empty=~all;

    Bitboard allPawns=pos->pieces[PAWN];
    Bitboard allKnights=pos->pieces[KNIGHT];
    Bitboard allBishops=pos->pieces[BISHOP];
    Bitboard allQueens=pos->pieces[QUEEN];

    Bitboard friendlyPawns=allPawns&friendly;

    Bitboard enemyPawns=allPawns&enemy;
    Bitboard enemyBishops=allBishops&enemy;
    Bitboard enemyQueens=allQueens&enemy;

    Bitboard knights=allKnights&friendly;
    Bitboard friendlyKnights=knights;
    Bitboard bishops=allBishops&friendly;
    Bitboard friendlyBishops=bishops;
    Bitboard rooks=pos->pieces[ROOK]&friendly;
    Bitboard queens=allQueens&friendly;
    Bitboard friendlyQueens=queens;
    Bitboard kings=pos->pieces[KING];
    Bitboard king=kings&friendly;


    //probably doesnt help much-
    int infiltration=0;
    Bitboard enemyHalf=opp?whiteSide:blackSide;
    infiltration+=popcount(friendlyPawns&enemyHalf);
    infiltration+=popcount(knights&enemyHalf)<<1;
    infiltration+=popcount(bishops&enemyHalf);
    infiltration+=popcount(rooks&enemyHalf)*3;
    infiltration+=popcount(queens&enemyHalf)*3;
    infiltration+=popcount(king&enemyHalf)*3;
    infiltration*=infiltration;
    score+=makeScore((infiltration/2)/3,infiltration);

    
    Sq kingSq=counttrailingzeros(king);
    Sq oppKingSq=counttrailingzeros(kings&enemy);
    //initalize kings tropisms:
    //tropism needs to be computed for per-side king, but here we have only the per-side pawns. do it from a different perspective: tropism is added when our things are close to our king, subtracted when our things are close to the opponent's king
    int kingPawnTropism=0;
    int kingPieceTropism=0;

    //their king zone is ring1 + 3 sqs a rank up towards the opponent(us)
    Bitboard theirKingZone=kingAttacks[oppKingSq];
    theirKingZone|=side?theirKingZone>>8:theirKingZone<<8;
    //count our total attacks units in their king zone
    int kingZoneAttackUnits=0;

    int colorPieceOffset=opp*6;//aka side?0:6

    //pawn evals
    Bitboard pawns=allPawns&friendly;

    //the squares that enemy pawns are capable of blocking/attacking now and in the future
    Bitboard adjEnemyPawns=shiftLeft(enemyPawns)|shiftRight(enemyPawns);
    Bitboard enemyPawnPossibleSights=enemyPawns|adjEnemyPawns;
    enemyPawnPossibleSights=side?fillDown(enemyPawnPossibleSights>>8):fillUp(enemyPawnPossibleSights<<8);

    //get ready to compute the attack masks
    Bitboard ourAttacks=0;
    Bitboard blockers=all;
    Bitboard enemyPawnAttacks=side?adjEnemyPawns>>8:adjEnemyPawns<<8;

    Bitboard enemyPawnPossibleSightAttacks=side?fillDown(enemyPawnAttacks):fillUp(enemyPawnAttacks);
    //squares on the 4th, 5th, 6th ranks(from white perspective) where enemy pawns cant kick
    Bitboard outpostMask=(~enemyPawnPossibleSightAttacks)&(side?rank456:rank543);

    Bitboard pawnSideFiles=shiftLeft(pawns)|shiftRight(pawns);
    
    //pawn stuff in bulk
    Bitboard ourDoubledPawns=(side?pawns>>8:pawns<<8)&pawns;//if a pawn has friendly pawn 1 square in front
    Bitboard ourPassedPawns=pawns&(~enemyPawnPossibleSights);
    Bitboard ourIsolatedPawns=pawns&(~fillUp(fillDown(pawnSideFiles)));
    Bitboard ourConnectedPawns=pawns&(pawnSideFiles|(pawnSideFiles<<8)|(pawnSideFiles>>8));
    Bitboard backwardsCantAdvanceTo=enemyPawnAttacks|enemyPawns;
    Bitboard ourBackwardPawns=pawns&(side?(backwardsCantAdvanceTo>>8)&(~fillUp(pawnSideFiles)):(backwardsCantAdvanceTo<<8)&(~fillDown(pawnSideFiles)));//if the pawn cant advance(square forwards is attacked by enemy pawns or is blocked by an enemy pawn) and has no pawns behind it to support
    

    score+=doubledPawnsPenalty*popcount(ourDoubledPawns);
    score+=isolatedPawnPenalty*popcount(ourIsolatedPawns);
    score+=backwardsPawnPenalty*popcount(ourBackwardPawns);

    Bitboard passers=ourPassedPawns;
    while(passers){
        Sq sq=poplsb(passers);
        score+=passedPawnRankBonus[side?sq>>3:7-(sq>>3)];
    }
    while(ourConnectedPawns){
        Sq sq=poplsb(ourConnectedPawns);
        score+=connectedPawnRankBonus[side?sq>>3:7-(sq>>3)];
    }

    // trace.pawns[side]=score;

    //mobility area: sqs that count in the mobility bonus for non-pawn & non-king pieces
    //remove sqs of our pawns that are on their starting 2 ranks or that are blocked
    //remove sqs of our king or queen
    //remove sqs attacked by enemy pawns
    Bitboard pawnsBlockingMobility=pawns&(side?((all>>8)|rank23):((all<<8)|rank76));
    Bitboard mobilityArea=~(pawnsBlockingMobility|king|queens|enemyPawnAttacks);
    

    while(pawns){
        Sq sq=poplsb(pawns);
        score+=pst[colorPieceOffset][sq];
        // trace.pst[side]+=pst[colorPieceOffset][sq];

        int isPassed=(ourPassedPawns&(1ull<<sq))>0;
        kingPawnTropism+=kingPawnTropisms[distances[sq][kingSq]]<<isPassed;
        kingPawnTropism-=kingPawnTropisms[distances[sq][oppKingSq]]<<isPassed;
    }

    while(knights){
        Sq sq=poplsb(knights);
        score+=pst[colorPieceOffset+1][sq];
        // trace.pst[side]+=pst[colorPieceOffset+1][sq];

        Bitboard sqBB=1ull<<sq;
        score+=knightOutpostBonus*((sqBB&outpostMask)>0);
        // trace.outpost[side]+=knightOutpostBonus*((sqBB&outpostMask)>0);

        Bitboard attacks=knightAttacks[sq];
        ourAttacks|=attacks;
        score+=MobilityBonus[0][popcount(attacks&mobilityArea)];
        // trace.mobility[side]+=MobilityBonus[0][popcount(attacks&mobilityArea)];

        kingPieceTropism+=kingPieceTropisms[distances[sq][kingSq]]*knightTropism>>1;
        kingPieceTropism+=kingPieceTropisms[distances[sq][oppKingSq]]*knightTropism;

        kingZoneAttackUnits+=popcount(attacks&theirKingZone)*minorAttackUnits;
    }

    if(popcount(bishops)>1){
        score+=bishopPairBonus;
        // trace.bishops[side]+=bishopPairBonus;
    }
    //find all friendly pawns that are physically blocked by an enemy pawn
    Bitboard blockadedFriendlyPawns=(side?enemyPawns>>8:enemyPawns<<8)&friendlyPawns;
    //bad bishops: penalize per friendly pawn of the same color, 2x if that pawn is blocked
    score+=((bishops&LIGHT_SQUARES)>0)*(popcount(LIGHT_SQUARES&friendlyPawns)+popcount(LIGHT_SQUARES&blockadedFriendlyPawns))*badBishopPerPawnPenalty;
    score+=((bishops&DARK_SQUARES)>0)*(popcount(DARK_SQUARES&friendlyPawns)+popcount(DARK_SQUARES&blockadedFriendlyPawns))*badBishopPerPawnPenalty;
    // trace.bishops[side]+=((bishops&LIGHT_SQUARES)>0)*(popcount(LIGHT_SQUARES&friendlyPawns)+popcount(LIGHT_SQUARES&blockadedFriendlyPawns))*badBishopPerPawnPenalty;
    // trace.bishops[side]+=((bishops&DARK_SQUARES)>0)*(popcount(DARK_SQUARES&friendlyPawns)+popcount(DARK_SQUARES&blockadedFriendlyPawns))*badBishopPerPawnPenalty;

    while(bishops){
        Sq sq=poplsb(bishops);
        score+=pst[colorPieceOffset+2][sq];
        // trace.pst[side]+=pst[colorPieceOffset+2][sq];
        
        Bitboard sqBB=1ull<<sq;
        score+=bishopOutpostBonus*((sqBB&outpostMask)>0);
        // trace.outpost[side]+=bishopOutpostBonus*((sqBB&outpostMask)>0);
        
        Bitboard attacks=getBishopAttacks(sq,blockers);
        ourAttacks|=attacks;
        score+=MobilityBonus[1][popcount(attacks&mobilityArea)];
        // trace.mobility[side]+=MobilityBonus[1][popcount(attacks&mobilityArea)];

        kingPieceTropism+=kingPieceTropisms[distances[sq][kingSq]]*bishopTropism>>1;
        kingPieceTropism+=kingPieceTropisms[distances[sq][oppKingSq]]*bishopTropism;
        
        kingZoneAttackUnits+=popcount(attacks&theirKingZone)*minorAttackUnits;
    }

    while(rooks){
        Sq sq=poplsb(rooks);
        score+=pst[colorPieceOffset+3][sq];
        // trace.pst[side]+=pst[colorPieceOffset+3][sq];
        
        Bitboard thisFile=fileA<<(sq&7);
        score+=((thisFile&friendlyPawns)<1)*((thisFile&enemyPawns)?semiOpenFileRook:openFileRook);
        // trace.rookFiles[side]+=((thisFile&friendlyPawns)<1)*((thisFile&enemyPawns)?semiOpenFileRook:openFileRook);

        Bitboard attacks=getRookAttacks(sq,blockers);
        ourAttacks|=attacks;
        score+=MobilityBonus[2][popcount(attacks&mobilityArea)];
        // trace.mobility[side]+=MobilityBonus[2][popcount(attacks&mobilityArea)];
        
        kingPieceTropism+=kingPieceTropisms[distances[sq][kingSq]]*rookTropism>>1;
        kingPieceTropism+=kingPieceTropisms[distances[sq][oppKingSq]]*rookTropism;
        
        kingZoneAttackUnits+=popcount(attacks&theirKingZone)*rookAttackUnits;
    }
    while(queens){
        Sq sq=poplsb(queens);
        score+=pst[colorPieceOffset+4][sq];
        // trace.pst[side]+=pst[colorPieceOffset+4][sq];

        Bitboard attacks=(getBishopAttacks(sq,blockers)|getRookAttacks(sq,blockers));
        ourAttacks|=attacks;
        score+=MobilityBonus[3][popcount(attacks&mobilityArea)];
        // trace.mobility[side]+=MobilityBonus[3][popcount(attacks&mobilityArea)];

        kingPieceTropism+=kingPieceTropisms[distances[sq][kingSq]]*queenTropism>>1;
        kingPieceTropism+=kingPieceTropisms[distances[sq][oppKingSq]]*queenTropism;
        
        kingZoneAttackUnits+=popcount(attacks&theirKingZone)*queenAttackUnits;
    }

    score+=pst[colorPieceOffset+5][kingSq];

    //stuff for king safety
    int kingSafety=0;//centipawn ish

    int kingFile=kingSq&7;

    Bitboard kingFileC=fileA<<kingFile;
    Bitboard kingFileL=shiftLeft(kingFileC);
    Bitboard kingFileR=shiftRight(kingFileC);

    Bitboard kingFront=kingFileL|kingFileC|kingFileR;
    int rankShift=kingSq&18446744073709551608ull;
    kingFront=side?(kingFront<<(rankShift+8)):(kingFront>>(64-rankShift));
    
    Bitboard kingRing=kingAttacks[kingSq];
    ourAttacks|=kingRing;

    //the 3x1 region in front of the king, computed hackily
    Bitboard pawnShield1=kingFront&kingRing;
    //3x1 region 2 ranks in front of the king
    Bitboard pawnShield2=side?pawnShield1<<8:pawnShield1>>8;

    //pawn shields
    kingSafety+=popcount(pawnShield1&friendlyPawns)*pawnShield1Safety;
    kingSafety+=popcount(pawnShield2&friendlyPawns)*pawnShield2Safety;

    //3 adjacent semi-open/open files: if a file has less than 2 pawns, its exposed (a cheap and low effort assumption). x1.5 weight for king's direct file
    kingSafety+=((popcount(kingFileC&allPawns)<2)*kingExposedFileSafety*3)>>1;
    kingSafety+=(popcount(kingFileL&allPawns)<2)*kingExposedFileSafety;
    kingSafety+=(popcount(kingFileR&allPawns)<2)*kingExposedFileSafety;

    //check for king's exposed diagonals as if it were a bishop, penalize the mobility: the blockers are friendly pieces or enemy pawns. dampen x0.5 if theres no matching colored bishop to attack
    Bitboard kingDiagBlockers=friendly|enemyPawns;
    Bitboard kingColorMask=king&DARK_SQUARES?DARK_SQUARES:LIGHT_SQUARES;
    int bishopDampen=(enemyBishops&kingColorMask)>0;
    kingSafety+=(popcount(getBishopAttacks(kingSq,kingDiagBlockers))*kingDiagonalSpaceSafety)>>(bishopDampen);

    //king safety matters less if: has castling rights, or enemy queen gone. x0.5 each
    kingSafety=kingSafety>>(((pos->castling&(side?0b1100:0b0011))>0)+(enemyQueens<1));

    score+=makeScore(kingSafety,0);
    // trace.kingSafety[side]+=makeScore(kingSafety,0);

    score+=makeScore(kingPieceTropism,kingPawnTropism);

    //penalty per undefended minor
    score+=undefendedMinorPenalty*popcount((~ourAttacks)&(friendlyKnights|friendlyBishops));
    
    //our attack units for their king zone
    score+=makeScore(SafetyTable[kingZoneAttackUnits]>>int(friendlyQueens<1||enemyQueens>1),0);


    pawns=friendlyPawns;
    //generate our pawn pushes
    Bitboard pawnPush=(side?pawns<<8:pawns>>8)&empty;
    Bitboard doublePawnPush=(side?(pawnPush<<8)&rank4:(pawnPush>>8)&rank5)&empty;
    //sqs that can be attacked by our pawns in 1 move. the pushes should be somewhat safe
    Bitboard ourPawnThreats=(pawnPush|doublePawnPush)&ourAttacks&(~enemyPawnAttacks);
    Bitboard pawnThreatEast=side?(ourPawnThreats&not_fileH)<<9:(ourPawnThreats&not_fileH)>>7;
    Bitboard pawnThreatWest=side?(ourPawnThreats&not_fileA)<<7:(ourPawnThreats&not_fileA)>>9;
    ourPawnThreats=pawnThreatEast|pawnThreatWest;
    //bonus for attacked pieces
    int pawnThreats=popcount(ourPawnThreats&enemy);
    score+=makeScore(pawnThreats*pawnThreatBonusMG,pawnThreats*pawnThreatBonusEG);

    // trace.sum[side]=score;
    return score;
}


int evaluate(Position *pos);

//for null move pruning
bool hasNonPawnMaterial(Position *pos,Side side){
    return (pos->colors[side]&(~pos->pieces[PAWN])&(~pos->pieces[KING])) >0;
}


constexpr int seeValues[6]={100, 300, 300, 500, 900, 20000};

static inline Bitboard leastValuableAttacker(const Position *pos, Bitboard attackers, Side side, Piece &outType){
    Bitboard ours=attackers&pos->colors[side];
    for(Piece pc=PAWN; pc<=KING; pc++){
        Bitboard bb=ours&pos->pieces[pc];
        if(bb){
            outType=pc;
            return bb&-bb;
        }
    }
    return 0;
}

int SEE(Position *pos, Move move){
    int gain[32],d=0;

    Sq from=move&63,to=(move>>6)&63;
    int flag=move>>14;
    bool EP=flag==FLAG_EN_PASSANT;

    Piece pieceCaptured=pos->board[to];
    bool isCapture=pieceCaptured!=0||EP;

    //what the first move captured
    gain[0]=isCapture?seeValues[EP?PAWN:(pieceCaptured-1)%6]:0;

    Piece attacker=(pos->board[from]-1)%6;

    Bitboard occupancy=pos->colors[WHITE]|pos->colors[BLACK];
    Bitboard fromBB=1ull<<from;
    Bitboard attackers=sqAttackers(pos,to);

    Bitboard queens=pos->pieces[QUEEN];
    Bitboard diags=pos->pieces[BISHOP]|queens;
    Bitboard straights=pos->pieces[ROOK]|queens;

    if(EP){
        Sq epSq=to+(pos->turn==WHITE?-8:8);
        occupancy^=(1ull<<epSq);
    }

    Side side=pos->turn;

    do {
        d++;
        side^=1;

        gain[d]=seeValues[attacker]-gain[d-1];

        attackers^=fromBB;
        occupancy^=fromBB;

        if(attacker==PAWN || attacker==BISHOP || attacker==QUEEN)
            attackers|=getBishopAttacks(to,occupancy)&diags;
        if(attacker==ROOK || attacker==QUEEN)
            attackers|=getRookAttacks(to,occupancy)&straights;
        attackers&=occupancy;

        fromBB=leastValuableAttacker(pos,attackers,side,attacker);

        if(fromBB && attacker==KING&&(attackers&pos->colors[side^1]))
            fromBB=0;

    }while(fromBB);

    while(--d)
        gain[d-1]=-max(-gain[d-1], gain[d]);

    return gain[0];
}

//if SEE is equal or better than threshold
bool SEE_ge(Position *pos, Move move, int threshold){

    Sq from=move&63,to=(move>>6)&63;
    int flag=move>>14;
    bool EP=flag==FLAG_EN_PASSANT;

    Piece pieceCaptured=pos->board[to];
    bool isCapture=pieceCaptured!=0 || EP;
    Piece attacker=(pos->board[from]-1)%6;

    int balance=(isCapture?seeValues[EP?PAWN:(pieceCaptured-1)%6]:0)-threshold;

    if(balance<0)
        return false;

    balance-=seeValues[attacker];
    if(balance>=0)
        return true;

    Bitboard occupancy=pos->colors[WHITE]|pos->colors[BLACK];
    Bitboard fromBB=1ull << from;
    Bitboard attackers=sqAttackers(pos,to);

    Bitboard queens=pos->pieces[QUEEN];
    Bitboard diags=pos->pieces[BISHOP]|queens;
    Bitboard straights=pos->pieces[ROOK]|queens;

    if(EP){
        Sq epSq=to+(pos->turn==WHITE?-8:8);
        occupancy^=(1ull<<epSq);
    }

    attackers^=fromBB;
    occupancy^=fromBB;
    if(attacker==PAWN || attacker==BISHOP || attacker==QUEEN)
        attackers|=getBishopAttacks(to, occupancy)&diags;
    if(attacker==ROOK || attacker==QUEEN)
        attackers|=getRookAttacks(to, occupancy)&straights;
    attackers &= occupancy;

    Side side=pos->turn^1;

    while (true){
        fromBB=leastValuableAttacker(pos,attackers,side,attacker);
        if(!fromBB)
            break;

        attackers^=fromBB;
        occupancy^=fromBB;
        if(attacker==PAWN || attacker==BISHOP || attacker==QUEEN)
            attackers|=getBishopAttacks(to, occupancy)&diags;
        if(attacker==ROOK || attacker==QUEEN)
            attackers|=getRookAttacks(to, occupancy)&straights;
        attackers&=occupancy;

        side^=1;
        balance=-balance-1-seeValues[attacker];

        if(balance>=0){
            if(attacker==KING && (attackers & pos->colors[side]))
                side^=1;
            break;
        }
    }

    return pos->turn!=side;
}


constexpr int moveLevel=100000;

//with gravity or whatever
inline void applyHistoryBonus(int &history,int bonus){
    history+=bonus-history*abs(bonus)/16384;
}
void applyContHistBonus(Position* pos,UndoState* undoPtr,int ind,int bonus){
    if(pos->ply>=1 && (undoPtr-1)->move)
        applyHistoryBonus((undoPtr-1)->continuationHistory[ind],bonus);
    if(pos->ply>=2 && (undoPtr-2)->move)
        applyHistoryBonus((undoPtr-2)->continuationHistory[ind],(bonus*110)/128);
    if(pos->ply>=4 && (undoPtr-4)->move)
        applyHistoryBonus((undoPtr-4)->continuationHistory[ind],(bonus*90)/128);
    if(pos->ply>=6 && (undoPtr-6)->move)
        applyHistoryBonus((undoPtr-6)->continuationHistory[ind],(bonus*65)/128);
} 


inline void applyCorrHistBonus(int &history,int bonus){
    history+=bonus-history*abs(bonus)/8192;
}

inline int correctionValue(int pawn,int nonpawn0,int nonpawn1,int minor,int major){
    return (pawn*310+(nonpawn0+nonpawn1)*130+minor*230+major*194)/(32768);
}

inline bool isMate(int score){
    return abs(score)>MATE_THRESHOLD;
}

int Worker::scoreMove(Position *pos,Move move,Move countermove){

    Sq from=move&63,to=(move>>6)&63;
    int flag=move>>14;
    Piece pieceMoving=pos->board[from];
    Piece captured=flag==FLAG_EN_PASSANT?(pos->turn?PAWN_B:PAWN_W):pos->board[to];

    int score=0;
    if(flag==FLAG_PROMOTION)score+=30000;

    if(captured){
        int see=SEE(pos,move);
        int ind=(pieceMoving-1)*64+to;

        score+=16384;
        score+=captureHistoryTable[ind][(captured-1)%6];
        
        score=min(max(score,0),moveLevel);

        if(see>=0)
            score+=moveLevel*(see/100+1);
        else
            score-=moveLevel*(-see/100);

        return score;
    }else{

        int dbPly=pos->ply<<1;

        int ind=(pieceMoving-1)*64+to;
        score+=16384*3;
        score+=historyTable[ind];

        if(move==killerMoves[dbPly])score+=60000;
        if(move==killerMoves[dbPly+1])score+=60000;

        if(move==countermove)score+=6000;

        if(pos->ply>=1 && (undoPtr-1)->move)score+=((undoPtr-1)->continuationHistory[ind]*70)/128;
        if(pos->ply>=2 && (undoPtr-2)->move)score+=((undoPtr-2)->continuationHistory[ind]*60)/128;
        if(pos->ply>=4 && (undoPtr-4)->move)score+=((undoPtr-4)->continuationHistory[ind]*50)/128;
        if(pos->ply>=6 && (undoPtr-6)->move)score+=((undoPtr-6)->continuationHistory[ind]*40)/128;
        
        score=min(max(score,0),moveLevel);
        return score;
    }

    return score;
}

void Worker::scoreMoves(Position *pos,Move* startMoveList,Move* endMoveList,Move ttMove){

    Move countermove=0;
    if(pos->ply>0){
        Move moveBefore=(undoPtr-1)->move;
        Piece pieceBefore=(undoPtr-1)->moved;
        Sq toBefore=(moveBefore>>6)&63;
        int indBefore=(pieceBefore-1)*64+toBefore;
        countermove=countermoveHistory[indBefore];
    }

    for(Move* move=startMoveList;move<endMoveList;move++){
        moveScores[move-moveList]=*move==ttMove?2000000000:scoreMove(pos,*move,countermove);
    }
}

void Worker::pickMove(Move* move,Move* endMoveList){
    
    int bestScore=-999999;
    int startIndex=move-moveList;
    int bestIndex=startIndex;

    for(Move* m=move;m<endMoveList;m++){
        int moveScore=moveScores[m-moveList];
        if(moveScore>bestScore){
            bestScore=moveScore;
            bestIndex=m-moveList;
        }
    }

    std::swap(moveList[bestIndex],moveList[startIndex]);
    std::swap(moveScores[bestIndex],moveScores[startIndex]);
}

int Worker::qsearch(Position *pos,int alpha,int beta,bool pvNode=false){

    if(!(nodeCount&4095) || searchTimeStopped){
        out_nodes.store(nodeCount,std::memory_order_relaxed);
        if((GetTickCount64()>searchTimeLimit || globalStopFlag.load(std::memory_order_acquire)) || searchTimeStopped){
            searchTimeStopped=true;
            return 0;
        }
    }

    if(isDrawByRule(undoPtr,pos))
        return 0;

    Move bestMove=0;
    int originalAlpha=alpha;

    TTEntry entry=TT[pos->key&TTSizeAnd];
    Move ttMove=0;
    int ttScore=0;
    bool ttHit=false;

    if(entry.key==pos->key){

        ttMove=bestMove=entry.bestMove;
        ttScore=entry.score;
        ttHit=true;
        
        //fix mate distance
        if(ttScore>MATE_THRESHOLD)ttScore-=pos->ply;
        if(ttScore<-MATE_THRESHOLD)ttScore+=pos->ply;

        if(!pvNode){

            if(entry.flag==EXACT)
                return ttScore;

            if(entry.flag==LOWER && ttScore>=beta)
                return ttScore;

            if(entry.flag==UPPER && ttScore<=alpha)
                return ttScore;
        }
    }

    //set up check info
    Sq kingSq=counttrailingzeros(pos->pieces[KING]&pos->colors[pos->turn]);
    Bitboard checkers=sqAttackers(pos,kingSq,pos->turn^1);
    int checkDegree=popcount(checkers);

    
    //if not in check: gen all caps. otherwise, evasions need to be looked at too
    Move* startMoveList=movePtr;
    int bestScore=-INF;
    int rawStaticEval=evaluate(pos);
    if(!checkDegree){

        bestScore=rawStaticEval;
        
        int pawnCorrHist=pawnCorrectionHistory[pos->turn][pos->pawnKey&corrHistSizeAnd];
        int nonpawn0CorrHist=nonpawnCorrectionHistory[pos->turn][0][pos->nonPawnKey[0]&corrHistSizeAnd];
        int nonpawn1CorrHist=nonpawnCorrectionHistory[pos->turn][1][pos->nonPawnKey[1]&corrHistSizeAnd];
        int minorCorrHist=minorCorrectionHistory[pos->turn][pos->minorKey&corrHistSizeAnd];
        int majorCorrHist=majorCorrectionHistory[pos->turn][pos->majorKey&corrHistSizeAnd];

        bestScore+=correctionValue(pawnCorrHist,nonpawn0CorrHist,nonpawn1CorrHist,minorCorrHist,majorCorrHist);
        
        if(ttHit)
            if(entry.flag==EXACT||(entry.flag==LOWER&&ttScore>bestScore)||(entry.flag==UPPER&&ttScore<bestScore))
                bestScore=ttScore;

        //stand pat
        if(bestScore>=beta)
            return bestScore;
        
        //delta pruning
        if(bestScore+900<alpha)
            return alpha;

        if(bestScore>alpha)
            alpha=bestScore;
        
        generateLegalCaptures(movePtr,pos,checkers,checkDegree);

    }else{

        generateLegalMoves(movePtr,pos,checkers,checkDegree);
        if(movePtr==startMoveList)
            return -INF+pos->ply;
    }
    Move* endMoveList=movePtr;
    
    scoreMoves(pos,startMoveList,endMoveList,ttMove);

    for(Move* move=startMoveList;move<endMoveList;move++){
        pickMove(move,endMoveList);

        //prune negative see captures
        if(!pvNode && !checkDegree && (!SEE_ge(pos,*move,0)))continue;

        makeMove(undoPtr,pos,*move,this);nodeCount++;
        int score=-qsearch(pos,-beta,-alpha,pvNode);
        undoMove(undoPtr,pos);

        if(score>bestScore){
            bestScore=score;
            bestMove=*move;
        }
        if(score>alpha)
            alpha=score;
        if(score>=beta)
            break;
    }
    movePtr=startMoveList;

    TTEntry old=TT[pos->key&TTSizeAnd];
    //write to TT
    if((0>=old.depth-(ttAge-old.age)+(old.flag==EXACT||abs(old.score)>MATE_THRESHOLD) || pvNode) && !searchTimeStopped && bestMove){

        TTEntry& write=TT[pos->key&TTSizeAnd];
        write.age=ttAge;
        write.key=pos->key;
        write.depth=0;
        write.score=bestScore;
        write.bestMove=bestMove;
        write.staticEval=rawStaticEval;
        //classify this node
        if(bestScore<=originalAlpha)
            write.flag=UPPER;
        else if(bestScore>=beta)
            write.flag=LOWER;
        else
            write.flag=EXACT;
        
        // adjust mate scores before storing
        if(write.score>MATE_THRESHOLD)
            write.score+=pos->ply;
        else if(write.score<-MATE_THRESHOLD)
            write.score-=pos->ply;
    }


    return bestScore;
}

int Worker::search(Position *pos,int depth,int alpha,int beta){

    if(!(nodeCount&4095) || searchTimeStopped){
        out_nodes.store(nodeCount,std::memory_order_relaxed);
        if((GetTickCount64()>searchTimeLimit || globalStopFlag.load(std::memory_order_acquire)) || searchTimeStopped){
            searchTimeStopped=true;
            return 0;
        }
    }

    if(isDrawByRule(undoPtr,pos))
        return 0;

    bool rootNode=pos->ply==0;
    bool pvNode=beta>alpha+1;
    
    if(depth<=0)
        return qsearch(pos,alpha,beta,pvNode);
        // return evaluate(pos);

    //mate distance pruning
    alpha=max(-INF+pos->ply,alpha);
    beta=min(INF-(pos->ply+1),beta);
    if(alpha>=beta)return alpha;
    
    int originalAlpha=alpha;
    int bestScore=-INF;
    Move bestMove=0;

    //probe tt
    TTEntry entry=TT[pos->key&TTSizeAnd];
    Move ttMove=0;
    int ttScore=0;
    bool ttHit=false;

    if(entry.key==pos->key){

        ttMove=entry.bestMove;
        ttScore=entry.score;
        ttHit=true;
        
        //fix mate distance
        if(ttScore>MATE_THRESHOLD)ttScore-=pos->ply;
        if(ttScore<-MATE_THRESHOLD)ttScore+=pos->ply;

        //dont do TT cutoff if the depth is lower. dont use tt cutoffs when halfmoves is high, may miss repetitions. no tt cutoff in pv nodes
        if(entry.depth>=depth && pos->halfmoves<96 && !pvNode && !undoPtr->excludedMove){

            if(entry.flag==EXACT)
                return ttScore;

            if(entry.flag==LOWER && ttScore>=beta)
                return ttScore;

            if(entry.flag==UPPER && ttScore<=alpha)
                return ttScore;
        }
    }

    //set up check info
    Sq kingSq=counttrailingzeros(pos->pieces[KING]&pos->colors[pos->turn]);
    Bitboard checkers=sqAttackers(pos,kingSq,pos->turn^1);
    int checkDegree=popcount(checkers);

    int rawStaticEval=evaluate(pos);
    int staticEval=rawStaticEval;
    bool improving=false;

    int& pawnCorrHist=pawnCorrectionHistory[pos->turn][pos->pawnKey&corrHistSizeAnd];
    int& nonpawn0CorrHist=nonpawnCorrectionHistory[pos->turn][0][pos->nonPawnKey[0]&corrHistSizeAnd];
    int& nonpawn1CorrHist=nonpawnCorrectionHistory[pos->turn][1][pos->nonPawnKey[1]&corrHistSizeAnd];
    int& minorCorrHist=minorCorrectionHistory[pos->turn][pos->minorKey&corrHistSizeAnd];
    int& majorCorrHist=majorCorrectionHistory[pos->turn][pos->majorKey&corrHistSizeAnd];

    if(checkDegree){
        staticEval=pos->ply>=2?(undoPtr-2)->staticEval:staticEval;
        undoPtr->staticEval=staticEval;
    }else{

        staticEval+=correctionValue(pawnCorrHist,nonpawn0CorrHist,nonpawn1CorrHist,minorCorrHist,majorCorrHist);

        undoPtr->staticEval=staticEval;

        if(pos->ply>=2 && staticEval>(undoPtr-2)->staticEval)
            improving=true;

        //if a tt entry was probed, update static eval to the tt score, a better estimate
        if(ttHit)
            if(entry.flag==EXACT||(entry.flag==LOWER&&ttScore>staticEval)||(entry.flag==UPPER&&ttScore<staticEval))
                staticEval=ttScore;
    }

    //if not in check, root, or pv, do fun pruning
    if(!checkDegree && !rootNode && !pvNode && !undoPtr->excludedMove){
        
        bool tryNullMove=false;
        bool noMateScores=abs(beta)<MATE_THRESHOLD && abs(staticEval)<MATE_THRESHOLD && abs(alpha)<MATE_THRESHOLD;
        bool safeNullMoveDoIt=hasNonPawnMaterial(pos,pos->turn) && tryNullMove && noMateScores;

        //reverse futility pruning
        if(depth<=6 && staticEval-75*(depth-improving)-75>=beta && !isMate(beta))
            return staticEval;

        //razoring
        if(depth<=8 && staticEval<alpha-512-256*depth*depth && noMateScores){
            int razor=qsearch(pos,alpha,beta);
            if(razor<alpha)
                return razor;
        }

        //null move pruning
        if(depth>=3 && hasNonPawnMaterial(pos,pos->turn) && (undoPtr-1)->move!=0 && staticEval>=beta-depth*81+24 && !isMate(beta)){
            int R=3+(depth>>2)+min(max(staticEval-beta,0)/100,3);

            makeNullMove(undoPtr,pos);nodeCount++;
            int nullScore=-search(pos,depth-1-R,-beta,-beta+1);
            undoNullMove(undoPtr,pos);

            if(nullScore>=beta && !isMate(nullScore)){
                return nullScore;
            }
        }

        //probcut
        // int probBeta=beta+160-improving*25;
        // if(depth>=5 && abs(beta)<MATE_THRESHOLD && (!ttHit || entry.depth+3<depth || ttScore>probBeta)){
            
        //     Move* pcStart=movePtr;
        //     generateLegalCaptures(movePtr,pos,checkers,checkDegree);
        //     Move* pcEnd=movePtr;
        //     scoreMoves(pos,pcStart,pcEnd,ttMove);

        //     for(Move* m=pcStart;m<pcEnd;m++){
        //         pickMove(m,pcEnd);
        //         //negative SEE caps are scored <0
        //         if(moveScores[m-moveList]<0)continue;

        //         makeMove(undoPtr,pos,*m,this);nodeCount++;
        //         int s=-qsearch(pos,-probBeta,-probBeta+1);
                
        //         if(s>=probBeta)
        //             s=-search(pos,(depth-4)>>1,-probBeta,-probBeta+1);
                    
        //         undoMove(undoPtr,pos);

        //         if(s>=probBeta){
        //             movePtr=pcStart;
        //             return s;
        //         }
        //     }
        //     movePtr=pcStart;
        // }
    }

    //IIR
    if(depth>=4 && !checkDegree && !ttMove && !undoPtr->excludedMove)
        depth--;

    //move gen
    Move* startMoveList=movePtr;
    generateLegalMoves(movePtr,pos,checkers,checkDegree);
    Move* endMoveList=movePtr;

    //no legal moves: return checkmate or stalemate score
    if(startMoveList==endMoveList)
        return checkDegree?-INF+pos->ply:0;

    scoreMoves(pos,startMoveList,endMoveList,ttMove);

    int moveIndex=0;
    for(Move* move=startMoveList;move<endMoveList;move++){
        
        pickMove(move,endMoveList);

        if(*move==undoPtr->excludedMove)continue;

        Sq from=*move&63,to=(*move>>6)&63;
        int flag=(*move)>>14;
        Piece captured=flag==FLAG_EN_PASSANT?PAWN_W:pos->board[to];//will later get casted to (captured-1)%6, so color distinction doesnt matter
        bool wasCapture=captured;
        bool isBadCapture=!SEE_ge(pos,*move,0);
        int ind=(pos->board[from]-1)*64+to;

        //late move pruning
        if(!rootNode && !checkDegree && !pvNode && !wasCapture && moveIndex>depth*3+5+3*improving && depth<=3 && abs(beta)<MATE_THRESHOLD && alpha>-MATE_THRESHOLD)continue;

        //see pruning
        if(!rootNode && !checkDegree && !pvNode && depth<=8 && moveIndex>=2 && abs(beta)<MATE_THRESHOLD && alpha>-MATE_THRESHOLD && !SEE_ge(pos,*move,wasCapture?-90*depth:-50*depth))continue;

        //futility pruning
        if(!rootNode && !checkDegree && !pvNode && !wasCapture && depth<=3 && abs(beta)<MATE_THRESHOLD && alpha>-MATE_THRESHOLD && staticEval+90*depth+120<=alpha)continue;

        //history pruning
        if(!rootNode && !checkDegree && !pvNode && !wasCapture && moveScores[move-moveList]<(16384*3)-20*depth-60 && depth<=7 && abs(beta)<MATE_THRESHOLD && alpha>-MATE_THRESHOLD)continue;
        
        //singular extensions + double/negative extensions
        int E=0;
        if(!rootNode && *move==ttMove && !undoPtr->excludedMove && depth>=8 && (entry.flag==LOWER||entry.flag==EXACT) && abs(ttScore)<MATE_THRESHOLD && entry.depth>=depth-3){

            undoPtr->excludedMove=ttMove;
            int singBeta=ttScore-depth*2;
            int singScore=search(pos,(depth-1)>>1,singBeta-1,singBeta);
            undoPtr->excludedMove=0;

            if(singScore<singBeta){
                E=1+(!pvNode && singScore<singBeta-20);
            }else if(singScore>=beta && abs(singScore)<MATE_THRESHOLD){
                return singScore;
            }else if(ttScore>=beta){
                E-=3;
            }
        }


        int score=0;
        makeMove(undoPtr,pos,*move,this);nodeCount++;

        //PVS: search the first move (expected to be best after move ordering) with a full window, then search the rest with a null window
        if(moveIndex==0){
            score=-search(pos,depth-1+E,-beta,-alpha);
        }else{
            //null window search. integrate with late move reductions: dont lmr in pv, in check, or in non-bad captures
            if((!wasCapture||isBadCapture) && !checkDegree && !pvNode && alpha>-MATE_THRESHOLD && beta<MATE_THRESHOLD){

                int R=lmrTable[min(depth,63)][max(min(moveIndex,63),0)];
                R+=!improving;
                score=-search(pos,depth-1-R,-alpha-1,-alpha);

            }else
                score=-search(pos,depth-1,-alpha-1,-alpha);

            //if the null windows failed high, re-search at full depth and full window
            if(score>alpha)
                score=-search(pos,depth-1,-beta,-alpha);
        }


        undoMove(undoPtr,pos);

        if(score>bestScore){
            bestScore=score;
            bestMove=*move;
        }

        if(score>alpha)
            alpha=score;

        if(score>=beta){

            if(!wasCapture && !checkDegree && !undoPtr->excludedMove){
                int bonus=depth*depth;
                applyHistoryBonus(historyTable[ind],bonus);
                applyContHistBonus(pos,undoPtr,ind,bonus);

                for(int i=0;i<moveIndex;i++){
                    Move *_move=startMoveList+i;
                    Sq _from=*_move&63,_to=(*_move>>6)&63;
                    bool _wasCapture=pos->board[_to]!=0 || (*_move)>>14==FLAG_EN_PASSANT;
                    int _ind=(pos->board[_from]-1)*64+_to;
                    if(!_wasCapture){
                        applyHistoryBonus(historyTable[_ind],-bonus);
                        applyContHistBonus(pos,undoPtr,_ind,-bonus);
                    }
                }

                int dbPly=pos->ply<<1;
                killerMoves[dbPly+1]=killerMoves[dbPly];
                killerMoves[dbPly]=*move;

                if(pos->ply>0){
                    Move moveBefore=(undoPtr-1)->move;
                    Piece pieceBefore=(undoPtr-1)->moved;
                    Sq toBefore=(moveBefore>>6)&63;
                    int indBefore=(pieceBefore-1)*64+toBefore;
                    countermoveHistory[indBefore]=*move;
                }
            }
            if(wasCapture && !undoPtr->excludedMove){
                int bonus=depth*depth;
                applyHistoryBonus(captureHistoryTable[ind][(captured-1)%6],bonus);

                for(int i=0;i<moveIndex;i++){
                    Move *_move=startMoveList+i;
                    Sq _from=*_move&63,_to=(*_move>>6)&63;
                    int _flag=(*_move)>>14;
                    int _captured=_flag==FLAG_EN_PASSANT?PAWN_W:pos->board[_to];
                    int _ind=(pos->board[_from]-1)*64+_to;
                    if(_captured)
                        applyHistoryBonus(captureHistoryTable[_ind][(_captured-1)%6],-bonus);
                }
            }

            break;
        }
        
        moveIndex++;
    }
    movePtr=startMoveList;
    
    TTEntry old=TT[pos->key&TTSizeAnd];
    //store tt: overwrite everything, if tt entry is this node, overwrite if higher depth
    if((depth>=old.depth-(ttAge-old.age)+(old.flag==EXACT||abs(old.score)>MATE_THRESHOLD) || pvNode) && !searchTimeStopped && !undoPtr->excludedMove){
        
        TTEntry& write=TT[pos->key&TTSizeAnd];
        write.age=ttAge;
        write.key=pos->key;
        write.depth=depth;
        write.score=bestScore;
        write.bestMove=bestMove;
        write.staticEval=rawStaticEval;
        //classify this node
        if(bestScore<=originalAlpha)
            write.flag=UPPER;
        else if(bestScore>=beta)
            write.flag=LOWER;
        else
            write.flag=EXACT;
        
        // adjust mate scores before storing
        if(write.score>MATE_THRESHOLD)
            write.score+=pos->ply;
        else if(write.score<-MATE_THRESHOLD)
            write.score-=pos->ply;
    }

    //update correction history
    Sq to=(bestMove>>6)&63;
    bool wasCapture=pos->board[to] || (bestMove>>14)==FLAG_EN_PASSANT;
    if(!checkDegree && !undoPtr->excludedMove && abs(bestScore)<MATE_THRESHOLD){

        int bonus=(bestScore-staticEval)*depth/7;
        bonus=max(min(bonus,554),-554);
        applyCorrHistBonus(pawnCorrHist,bonus);
        // applyCorrHistBonus(nonpawn0CorrHist,bonus);
        // applyCorrHistBonus(nonpawn1CorrHist,bonus);
        // applyCorrHistBonus(minorCorrHist,(bonus*95)/128);
        // applyCorrHistBonus(majorCorrHist,(bonus*80)/128);
    }

    if(rootNode && !searchTimeStopped){
        out_bestMove.store(bestMove,std::memory_order_relaxed);
        out_bestScore.store(bestScore,std::memory_order_relaxed);
    }

    return bestScore;
}

void Worker::startSearch(){

    int searchDepth=64;
    int allotedTime=globalAllotedTime.load(std::memory_order_acquire);

    initSearch(&pos);

    Sq kingSq=counttrailingzeros(pos.pieces[KING]&pos.colors[pos.turn]);
    Bitboard checkers=sqAttackers(&pos,kingSq,pos.turn^1);
    int checkDegree=popcount(checkers);
    Move* startMoveList=movePtr;
    generateLegalMoves(movePtr,&pos,checkers,checkDegree);
    Move* endMoveList=movePtr;
    
    out_bestMove.store(0,std::memory_order_relaxed);

    if(startMoveList==endMoveList){
        out_bestMove.store(0,std::memory_order_release);
        out_bestScore.store(checkDegree?-INF:0,std::memory_order_release);
        return;
    }
    if(isDrawByRule(undoPtr,&pos)){
        out_bestMove.store(0,std::memory_order_release);
        out_bestScore.store(0,std::memory_order_release);
        return;
    }

    Move bestMove=0;
    int bestScore=-INF;
    int score=0;
    
    searchTimeLimit=GetTickCount64()+allotedTime;
    searchTimeStopped=false;

    //dont go past 2x the suggested time in competition
    uint64_t hardThinkingTimeLimit=searchTimeLimit+allotedTime;


    for(int depth=1+(id&1);depth<=searchDepth;depth++){

        //if depth is high enough to not be as noisy, do aspiration windows
        if(depth<=6)
            score=search(&pos,depth,-INF,INF);
        else{

            int aspirationDelta=25-(id&3)*2;
            int aspirationAlpha=score-aspirationDelta;
            int aspirationBeta=score+aspirationDelta;

            while(1){
                score=search(&pos,depth,aspirationAlpha,aspirationBeta);

                if(abs(score)>MATE_THRESHOLD)break;

                if(score<=aspirationAlpha){
                    aspirationAlpha=max(score-aspirationDelta,-INF);
                    aspirationDelta*=2;
                }else if(score>=aspirationBeta){
                    aspirationBeta=min(score+aspirationDelta,INF);
                    aspirationDelta*=2;
                }else{
                    break;
                }
            }
        }

        //if search was interrupted due to time limit, we don't need to discard it. the following lines read from the TT at the root, so if the TT entry for the root isn't wrong (not overwritten by a partial search), we can always use it

        //read both best move and score from the TT's root entry
        int bestScoreBefore=bestScore;
        Move bestMoveBefore=bestMove;
        bestMove=out_bestMove.load(std::memory_order_relaxed);
        bestScore=score;

        if(searchTimeStopped)
            break;

        //adjust thinking time based some factors
        if(!CLI){

            bool isWinning=bestScore>200;
            bool isNearMate=bestScore>800;
            bool hasMate=bestScore>MATE_THRESHOLD;

            //if best move changes (after low depth fluctuations)
            if(bestMoveBefore!=bestMove && depth>5)
                searchTimeLimit+=allotedTime/5;

            if(depth>5){
                int fluc=max(abs(bestScore-bestScoreBefore)-15,0);
                searchTimeLimit+=allotedTime*(fluc/200);
            }
               
            //if eval is lowering with depth and we're not very winning=we're worsening and realizing it
            if(bestScoreBefore<bestScore && !isWinning)
                searchTimeLimit+=allotedTime/15;
            
            //dont spend 2x more than the suggested time
            searchTimeLimit=min(searchTimeLimit,hardThinkingTimeLimit);
        }

        if(CLI && id==0){

            //thread 0 reports its own precise node count since its locally here. then sum the other threads
            int totalNodes=nodeCount;
            for(size_t i=1;i<allWorkers->size();i++)
                totalNodes+=(*allWorkers)[i].get()->out_nodes.load(std::memory_order_relaxed);

            std::cout << "depth " << depth << "  eval " << float(score*(pos.turn?1:-1))*0.01 << "  move " << algMove(movePtr,undoPtr,&pos,bestMove,this) << "  time " << (GetTickCount64()-(searchTimeLimit-allotedTime)) << "ms  nodes " << totalNodes << "  hashfull " << hashfull() << "  PV ";

            Position pvPos=pos;
            //PV from TT
            int safe=depth;
            Key repList[100];
            int repListSize=1;
            repList[0]=pos.key;
            while(safe--){

                TTEntry &e=TT[pvPos.key&TTSizeAnd];
                Sq kingSq=counttrailingzeros(pvPos.pieces[KING]&pvPos.colors[pvPos.turn]);
                Bitboard checkers=sqAttackers(&pvPos,kingSq,pvPos.turn^1);
                Move* startMoveList=movePtr;
                generateLegalMoves(movePtr,&pvPos,checkers,popcount(checkers));
                Move* endMoveList=movePtr;

                bool end=true;
                for(Move* move=startMoveList;move<endMoveList;move++){
                    if(*move==e.bestMove){
                        std::cout << algMove(movePtr,undoPtr,&pvPos,*move,this) << " ";
                        makeMove(undoPtr,&pvPos,*move,this);
                        repList[repListSize++]=pvPos.key;
                        undoPtr=undoStack;
                        end=false;
                        break;
                    }
                }
                movePtr=startMoveList;

                int c=0;
                for(int i=repListSize-3;i>=0;i-=2)
                    if(repList[i]==pvPos.key)
                        if(++c>=2){
                            end=true;
                            break;
                        }

                if(end)break;
            }

            std::cout << "\n";
        }
    }

    out_bestMove.store(out_bestMove.load(std::memory_order_relaxed),std::memory_order_release);
    out_bestScore.store(out_bestScore.load(std::memory_order_relaxed),std::memory_order_release);
}

struct UIState{
    Position pos;
    Move lastMove;
};
std::vector<UIState> UIHistory;

#if USE_NNUE

struct Network{
    Accumulator featureWeights[INPUT_SIZE];
    Accumulator featureBias;
    alignas(64) int16_t outputWeights[OUTPUT_BUCKETS][2*HIDDEN_SIZE];
    int16_t outputBias[OUTPUT_BUCKETS];
};


//"piece" is the [0,12] piece id (has color info)
inline int whiteFeatureIndex(Piece piece,Sq sq){
    return piece*64+sq;
}
inline int blackFeatureIndex(Piece piece,Sq sq){
    return (piece>5?piece-6:piece+6)*64+(sq^56);
}

Network net;

bool loadNNUE(const char* path){
    FILE* f=fopen(path,"r");
    if(!f){
        fprintf(stderr,"NNUE: could not open %s\n",path);
        return false;
    }

    float w;

    //W1: INPUT_SIZE*HIDDEN_SIZE floats then quantize by QA into featureWeights
    for(int i=0;i<INPUT_SIZE;i++){
        for(int j=0;j<HIDDEN_SIZE;j++){
            if(fscanf(f, "%f", &w)!=1){
                fprintf(stderr, "NNUE: read error at W1[%d][%d]\n", i, j);
                fclose(f);
                return false;
            }
            net.featureWeights[i].vals[j]=(int16_t)roundf(w*(float)QA);
        }
    }

    //b1: HIDDEN_SIZE floats then quantize by QA into featureBias
    for(int j=0;j<HIDDEN_SIZE;j++){
        if(fscanf(f,"%f",&w)!=1){
            fprintf(stderr,"NNUE: read error at b1[%d]\n",j);
            fclose(f);
            return false;
        }
        net.featureBias.vals[j]=(int16_t)roundf(w*(float)QA);
    }

    //W2: 2*HIDDEN_SIZE floats then quantize by QB into outputWeights
    for(int b=0;b<OUTPUT_BUCKETS;b++){
        for(int i=0;i<2*HIDDEN_SIZE;i++){
            if(fscanf(f,"%f",&w)!=1){
                fprintf(stderr,"NNUE: read error at W2[%d]\n",i);
                fclose(f);
                return false;
            }
            net.outputWeights[b][i]=(int16_t)roundf(w*(float)QB);
        }
    }

    //b2: 1 float then quantize by QA*QB into outputBias
    for(int b=0;b<OUTPUT_BUCKETS;b++){
        if(fscanf(f,"%f",&w)!=1){
            fprintf(stderr,"NNUE: read error at b2\n");
            fclose(f);
            return false;
        }
        net.outputBias[b]=(int16_t)roundf(w*(float)(QA*QB));
    }

    fclose(f);

    return true;
}


inline int32_t screlu(int16_t x){
    int32_t y=min(max((int32_t)x, 0),(int32_t)QA);
    return y*y;
}
inline __m256i clipped_relu16(__m256i x,__m256i zero,__m256i qa){
    return _mm256_min_epi16(_mm256_max_epi16(x,zero),qa);
}

int evaluateNNUE(Position *pos,Accumulator& us,Accumulator& them){

    Bitboard occ=pos->colors[WHITE]|pos->colors[BLACK];

    if(popcount(occ)==2)return 0;

    int bucket=getBucket(occ);
    
    //this is lizard screlu simd
    const __m256i zero=_mm256_setzero_si256();
    const __m256i qa=_mm256_set1_epi16(QA);
 
    __m256i acc64_lo=_mm256_setzero_si256();
    __m256i acc64_hi=_mm256_setzero_si256();
 
    for(int i=0;i<HIDDEN_SIZE;i+=16){
        //us accumulator
        __m256i va=_mm256_loadu_si256((const __m256i*)&us.vals[i]);
        __m256i wa=_mm256_loadu_si256((const __m256i*)&net.outputWeights[bucket][i]);
        __m256i ca=clipped_relu16(va,zero,qa);//v: range [0,QA]
        __m256i pa=_mm256_mullo_epi16(ca,wa);//v*w: range [0,QA*QB], fits in i16 if QA=255, QB=64, and w within range [-128,128] maybe with margin
        __m256i ta=_mm256_madd_epi16(pa,ca);//do the screlu: v*w*v
 
        //same but for them accumulator
        __m256i vb=_mm256_loadu_si256((const __m256i*)&them.vals[i]);
        __m256i wb=_mm256_loadu_si256((const __m256i*)&net.outputWeights[bucket][HIDDEN_SIZE+i]);
        __m256i cb=clipped_relu16(vb,zero,qa);
        __m256i pb=_mm256_mullo_epi16(cb,wb);
        __m256i tb=_mm256_madd_epi16(pb,cb);
 
        __m256i sum32=_mm256_add_epi32(ta,tb);//add and combine into i32
 
        //accumulate into sum
        acc64_lo=_mm256_add_epi64(acc64_lo,_mm256_cvtepi32_epi64(_mm256_castsi256_si128(sum32)));
        acc64_hi=_mm256_add_epi64(acc64_hi,_mm256_cvtepi32_epi64(_mm256_extracti128_si256(sum32,1)));
    }
 
    //get the 4 i64s out from avx2 and sum them
    __m256i acc64=_mm256_add_epi64(acc64_lo,acc64_hi);
    alignas(32) int64_t lanes[4];
    _mm256_storeu_si256((__m256i*)lanes,acc64);
    int64_t output=lanes[0]+lanes[1]+lanes[2]+lanes[3];
 
    //output is now quantized by QA*QA*QB
    int32_t out32=(int32_t)(output/(int64_t)QA);//remove a QA, now its QA*QB
    out32+=(int32_t)net.outputBias[bucket];//outputBias is in QA*QB scale, add now
    out32*=SCALE;//sigmoid scale thing from trainer
    out32/=(int32_t)(QA*QB);//remove remaining QA*QB, now its in centipawns
    // return out32;


    //without SIMD:
    // int64_t output=0;
    // int bucket=getBucket(pos->colors[WHITE]|pos->colors[BLACK]);
    // for(int i=0;i<HIDDEN_SIZE;i++)
    //     output+=(int64_t)screlu(us.vals[i])
    //            *(int64_t)net.outputWeights[bucket][i]+(int64_t)screlu(them.vals[i])
    //            *(int64_t)net.outputWeights[bucket][HIDDEN_SIZE+i];;

    // int32_t out32=(int32_t)(output / (int64_t)QA);
    // out32+=(int32_t)net.outputBias[bucket];
    // out32*=SCALE;
    // out32 /= (int32_t)(QA*QB);
    // return out32;

    // int knights=popcount(pos->pieces[KNIGHT]);
    // int bishops=popcount(pos->pieces[BISHOP]);
    // int rooks=popcount(pos->pieces[ROOK]);
    // int queens=popcount(pos->pieces[QUEEN]);
    // int sum=312*(knights+bishops)+512*rooks+912*queens;

    out32=out32*(8192+5000)/16384;

    // if(pos->ply>=1){
    //     Move last1=(undoPtr-1)->move;
    //     Move last2=pos->ply>=3?(undoPtr-3)->move:0;
    //     Move last3=pos->ply>=5?(undoPtr-5)->move:0;
    //     if((last1&63)==((last2>>6)&63) && (last2&63)==((last3>>6)&63))
    //         out32=out32*110/100;
    // }

    
    //label all KP vs K endgames as a draw except a few simple known winning cases and let search take the wheel 
    Bitboard pawns=pos->pieces[PAWN];
    if(popcount(pawns)==1 && !(pos->pieces[KNIGHT]|pos->pieces[BISHOP]|pos->pieces[ROOK]|pos->pieces[QUEEN])){

        //get sqs of the kings and pawn and reorient them to a global space so the attacking side is going to rank "8"
        Bitboard w=pos->colors[WHITE];
        int blackAttacking=((w&pawns)==0)*56;
        Sq pawn=poplsb(pawns)^blackAttacking;
        Bitboard kings=pos->pieces[KING];
        Bitboard att=kings&(blackAttacking?~w:w);
        Bitboard def=kings&(blackAttacking?w:~w);
        Sq attacker=poplsb(att)^blackAttacking;
        Sq defender=poplsb(def)^blackAttacking;
        bool attackerSTM=bool(blackAttacking)!=bool(pos->turn);
        bool flankPawn=(pawn&7)<1 || (pawn&7)>6;

        Sq promotingSq=56+(pawn&7);
        int stepsTillProm=7-(pawn>>3);
        //pawn double pushes and extra tempo
        if((pawn>>3)==1)stepsTillProm--;
        if(!attackerSTM)stepsTillProm++;

        //the rule of the square. if the cheb distance if too big, its immediately winning
        if(distances[defender][promotingSq]>stepsTillProm)
            return out32;

        //the textbook opposition position: if the defender needs to move when both kings are aligned in front of the pawn, its winning
        if(!flankPawn && attacker-8==pawn && defender-16==attacker && !attackerSTM)
            return out32;

        //if attacking king is >=6 rank and the pawn is just behind, there is a red carpet
        if(!flankPawn && (attacker>>3)>=5)
            if(attacker-8==pawn || attacker-9==pawn || attacker-7==pawn)
                return out32;

        return out32>>5;
    }


    return out32;
}



void addToNNUE(Position *pos,Piece piece,Sq sq){
    int wf=whiteFeatureIndex(piece,pos->mirrorWhite?(sq^7):sq);
    int bf=blackFeatureIndex(piece,pos->mirrorBlack?(sq^7):sq);
    pos->whiteAccumulator.add(net.featureWeights[wf].vals);
    pos->blackAccumulator.add(net.featureWeights[bf].vals);
}
void subFromNNUE(Position *pos,Piece piece,Sq sq){
    int wf=whiteFeatureIndex(piece,pos->mirrorWhite?(sq^7):sq);
    int bf=blackFeatureIndex(piece,pos->mirrorBlack?(sq^7):sq);
    pos->whiteAccumulator.sub(net.featureWeights[wf].vals);
    pos->blackAccumulator.sub(net.featureWeights[bf].vals);
}

void refreshWhiteAccumulator(Position *pos){
    bool mirror=(counttrailingzeros(pos->pieces[KING]&pos->colors[WHITE])&7)>=4;
    pos->mirrorWhite=mirror;

    pos->whiteAccumulator.reset(net.featureBias.vals);

    Bitboard all=pos->colors[WHITE]|pos->colors[BLACK];
    while(all){
        Sq sq=poplsb(all);
        Piece piece=pos->board[sq]-1;
        pos->whiteAccumulator.add(net.featureWeights[whiteFeatureIndex(piece,mirror?(sq^7):sq)].vals);
    }
}
void refreshBlackAccumulator(Position *pos){
    bool mirror=(counttrailingzeros(pos->pieces[KING]&pos->colors[BLACK])&7)>=4;
    pos->mirrorBlack=mirror;

    pos->blackAccumulator.reset(net.featureBias.vals);

    Bitboard all=pos->colors[WHITE]|pos->colors[BLACK];
    while(all){
        Sq sq=poplsb(all);
        Piece piece=pos->board[sq]-1;
        pos->blackAccumulator.add(net.featureWeights[blackFeatureIndex(piece,mirror?(sq^7):sq)].vals);
    }
}

void resetNNUE(Position *pos){
    refreshWhiteAccumulator(pos);
    refreshBlackAccumulator(pos);
}

#endif


int evaluate_HCE(Position *pos){

    Score score=evaluateSide(pos,WHITE)-evaluateSide(pos,BLACK);
    
    uint32_t phase=gamePhase(pos);
    // trace.phase=phase;
    Value mg=mgValue(score),eg=egValue(score);
    Value eval=lerpPhase(mg,eg,phase);


    // trace.beforeAdjust=eval;

    eval*=(pos->turn?1:-1);

    //tempo
    eval+=toMoveBonus;

    //drawish the closer the position is to rule 50
    eval=eval*(100-pos->halfmoves)/100;

    return (eval>>2)<<2;
}

#if USE_NNUE
int evaluate_NNUE(Position *pos){

    Accumulator& us=pos->turn?pos->whiteAccumulator:pos->blackAccumulator;
    Accumulator& them=pos->turn?pos->blackAccumulator:pos->whiteAccumulator;
    
    int eval=evaluateNNUE(pos,us,them);
    eval=eval*(200-pos->halfmoves)/200;
    return eval;
}
#endif

int evaluate(Position *pos){
    
    #if USE_NNUE
        return evaluate_NNUE(pos);
    #else
        return evaluate_HCE(pos);
    #endif
}


std::mt19937 RNG(std::random_device{}());
int randomInt(int min, int max){
    std::uniform_int_distribution<int> dist(min,max-1);
    return dist(RNG);
}


class ThreadPool{
public:

    alignas(64) std::vector<std::unique_ptr<Worker>> workers;

    ThreadPool(int numThreads){
        setThreadCount(numThreads);
    }

    void setThreadCount(int numThreads){
        shutdownThreads();

        workers.clear();
        nativeThreads.clear();

        exitFlag=false;
        searchGeneration=0;
        finishedCount=numThreads;

        for(int i=0;i<numThreads;i++){
            workers.push_back(std::make_unique<Worker>());
            workers[i]->id=i;
            nativeThreads.emplace_back(&ThreadPool::idleLoop, this, workers.back().get());
        }
        allWorkers=&workers;
    }

    ~ThreadPool(){
        shutdownThreads();
    }

    void startSearch(const Position& pos,int64_t allottedMs){

        globalAllotedTime.store(allottedMs,std::memory_order_release);
        globalStopFlag.store(false,std::memory_order_release);

        // give every worker its own copy of the position
        for(auto& w:workers)
            w->pos=pos;

        {
            std::lock_guard<std::mutex> lock(mtx);
            searchGeneration++;
            finishedCount=0;
        }

        startTime=GetTickCount64();
        cv.notify_all();
    }

    void clearHistories(){
        for(auto& w:workers)
            w->clearHistories();
    }

    //blocks until every worker has finished its startSearch
    void waitForFinish(){
        std::unique_lock<std::mutex> lock(mtx);
        cvDone.wait(lock,[this]{return finishedCount==workers.size();});
    }

    void stop(){
        globalStopFlag.store(true,std::memory_order_release);
        waitForFinish();
    }

    void doFinalResults(){
        Move bestMove=0;
        int bestScore=0;

        Worker* best=workers[0].get();
        // for(auto& w:workers){
        //     if(w->completedDepth>best->completedDepth || (w->completedDepth==best->completedDepth && w->out_bestScore.load(std::memory_order_acquire)>best->out_bestScore.load(std::memory_order_acquire))){
        //         best=w.get();
        //     }
        // }
        
        Worker* eng=workers[0].get();

        bestMove=eng->out_bestMove.load(std::memory_order_acquire);
        bestScore=eng->out_bestScore.load(std::memory_order_acquire);

        if(CLI){
            std::cout << "\nbestmove ";
            printMove(eng->movePtr,eng->undoPtr,&eng->pos,bestMove,eng);
            std::cout << "eval ";
            if(abs(bestScore)>MATE_THRESHOLD)
                std::cout<<(bestScore*(eng->pos.turn?1:-1)<0?"-":"+")<<"M"<<(INF-abs(bestScore)+1)/2;
            else
                std::cout<<float(bestScore*(eng->pos.turn?1:-1))*0.01;
            std::cout << "  stc " << float(evaluate(&eng->pos)*(eng->pos.turn?1:-1))*0.01 << "\n";

            int totalNodes=0;
            for(size_t i=0;i<allWorkers->size();i++)
                totalNodes+=(*allWorkers)[i].get()->out_nodes.load(std::memory_order_relaxed);
            
            int dur=GetTickCount64()-startTime;
            dur=max(dur,1);
            std::cout << "time " << dur << "ms  nodes " << totalNodes << "/" << float(totalNodes/dur)/1000 << "Mnps/" << float(totalNodes)/float(dur) << "Knps\n";
            
        }else{
            std::cout << "bestmove " << uciMove(&eng->pos,bestMove) << std::endl << std::flush;
            makeMove(eng->undoPtr,&eng->pos,bestMove,eng);
            updatePastPositionRepetition(&eng->pos);
        }
    }

private:

    std::vector<std::thread> nativeThreads;
    std::mutex mtx;
    std::condition_variable cv, cvDone;
    bool exitFlag=false;
    int  searchGeneration=0;
    size_t finishedCount=0;
    uint64_t startTime=0;

    void shutdownThreads(){
        if(nativeThreads.empty()) return;

        {
            std::lock_guard<std::mutex> lock(mtx);
            exitFlag=true;
        }

        cv.notify_all();
        for(auto& t:nativeThreads)t.join();

        nativeThreads.clear();
        workers.clear();
    }

    void idleLoop(Worker* w){
        int lastSeenGeneration=0;
        while(true){
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock,[&]{return exitFlag || searchGeneration!=lastSeenGeneration;});

            if(exitFlag)return;

            lastSeenGeneration=searchGeneration;
            lock.unlock();

            w->startSearch();

            lock.lock();
            finishedCount++;
            bool isLastWorker=(finishedCount==workers.size());
            lock.unlock();

            if(isLastWorker){
                doFinalResults();
                cvDone.notify_one();
            }
        }
    }
};

ThreadPool pool(THREADS);

int main(){
    
    #if USE_NNUE

        for(int i=0;i<INPUT_SIZE;i++)
            memset(net.featureWeights[i].vals,0,sizeof(net.featureWeights[i].vals));
        memset(net.outputWeights,0,sizeof(net.outputWeights));
        memset(net.featureBias.vals,0,sizeof(net.featureBias.vals));
        memset(net.outputBias,0,sizeof(net.outputBias));

        loadNNUE("nnue_512hl.txt");

    #endif

    std::cout << std::fixed << std::setprecision(2);

    initEngineTables();
    buildMagicAttackTable(true);
    buildMagicAttackTable(false);
    
    
    Worker* engine=pool.workers[0].get();

    initPosition(&engine->pos);



    //------------- perfting


    // initSearch(&pos);
    // clearTT();
    // printPosition(&pos);

    // auto t=std::chrono::high_resolution_clock::now();

    // int startDepth=6;
    // uint64_t nodes=perft(&pos,startDepth,startDepth);

    // auto end=std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> duration=end-t;

    // std::cout << "\n" << nodes << "\n";

    // std::cout << duration.count() << " ms\n";
    
    // return 0;


    //---------------- gen random opening book


    // std::ofstream file("noobyOpeningBook.txt");

    // for(int i=0;i<1000000;i++){
    //     initPosition(&pos);
    //     initSearch(&pos);

    //     for(int m=0;m<8;m++){
    //         movePtr=moveList;
    //         Sq kingSq=counttrailingzeros(pos.pieces[KING]&pos.colors[pos.turn]);
    //         Bitboard checkers=sqAttackers(&pos,kingSq,pos.turn^1);
    //         int checkDegree=popcount(checkers);
    //         Move* startMoveList=movePtr;
    //         generateLegalMoves(&pos,checkers,checkDegree);
    //         Move* endMoveList=movePtr;

    //         if(endMoveList-startMoveList==0)
    //             break;

    //         makeMove(&pos,*(startMoveList+randomInt(0,endMoveList-startMoveList)));

    //         file << getFen(&pos) << "\n";
    //     }

    //     if((i&32767)==0)
    //         std::cout << i << "\n";
    // }

    // file.close();
    // return 0;
    //-------------



    //not true uci: only meant for computer play. go cmds makes moves
    if(!CLI){

        std::string line;

        while(std::getline(std::cin,line)){

            if(line=="uci"){
                std::cout << "uciok\n" << std::flush;

            }else if(line=="quit"){
                return 0;

            }else if(line=="ucinewgame"){
                clearTT();
                pool.clearHistories();
            }else if(line=="isready"){
                std::cout << "readyok\n" << std::flush;
            }else if(line.rfind("position", 0)==0){

                std::stringstream ss(line);
                std::string token;

                ss>>token;//"position"
                ss>>token;

                if(token=="startpos"){
                    initPosition(&engine->pos);
                }else if(token=="fen"){

                    std::string fen;
                    std::string part;

                    //FEN has 6 space-separated fields
                    for(int i=0;i<6;i++){
                        ss>>part;
                        fen+=part+" ";
                    }

                    initFen(&engine->pos, fen.c_str());
                }

                //optional moves section
                if(ss>>token){

                    if(token=="moves"){

                        std::string moveStr;

                        while(ss>>moveStr){

                            Move move=stringToMove(&engine->pos, moveStr);

                            if(move==0){
                                std::cerr << "invalid move: " << moveStr << "\n";
                                break;
                            }

                            makeMove(engine->undoPtr,&engine->pos,move,engine);
                            updatePastPositionRepetition(&engine->pos);
                        }
                    }
                }
            }else if(line.rfind("go",0)==0){

                int wtime=0,btime=0,winc=0,binc=0;

                std::stringstream ss(line);
                std::string token;
                while(ss>>token){
                    if(token=="wtime")
                        ss>>wtime;
                    else if(token=="btime")
                        ss>>btime;
                    else if(token=="winc")
                        ss>>winc;
                    else if(token=="binc")
                        ss>>binc;
                }

                //think for 1s by default if no time remaining input is given
                int thinkTime=1000*20,thinkInc=0;

                if(engine->pos.turn==1){
                    if(wtime>0)
                        thinkTime=wtime;
                    if(winc>0)
                        thinkInc=winc;
                }else if(engine->pos.turn==0){
                    if(btime>0)
                        thinkTime=btime;
                    if(binc>0)
                        thinkInc=binc;
                }

                //recommended time
                thinkTime=thinkTime/(20)+thinkInc/2;

                pool.startSearch(engine->pos,thinkTime);

                //improper blocking for engine play uci
                pool.waitForFinish();
            }
        }
    }else{

        int CLI_MOVETIME=1000;

        while(1){

            std::cout << "\nTT: "<<(sizeof(TT)/(1024*1024))<<"MB\n";
            
            printPosition(&engine->pos);

            auto t=std::chrono::high_resolution_clock::now();

            pool.startSearch(engine->pos,CLI_MOVETIME);

            // pool.waitForFinish();

            bool invalidInputs=true;
            while(invalidInputs){

                std::string moveStr;
                std::getline(std::cin,moveStr);

                if(moveStr=="q"||moveStr=="quit"){return 0;}
                if(moveStr=="cleartt"){
                    clearTT();
                    pool.clearHistories();
                    break;
                }
                if(moveStr=="u"||moveStr=="undo"){
                    pool.stop();
                    if(UIHistory.size()>0){
                        engine->pos=UIHistory.back().pos;
                        UIHistory.pop_back();
                    }else{
                        std::cout << "nothing to undo\n";
                        continue;
                    }
                    break;
                }
                if(moveStr=="p"||moveStr=="printmoves"){
                    Sq kingSq=counttrailingzeros(engine->pos.pieces[KING]&engine->pos.colors[engine->pos.turn]);
                    Bitboard checkers=sqAttackers(&engine->pos,kingSq,engine->pos.turn^1);
                    int checkDegree=popcount(checkers);
                    Move* startMoveList=engine->movePtr;
                    generateLegalMoves(engine->movePtr,&engine->pos,checkers,checkDegree);
                    Move* endMoveList=engine->movePtr;
                    for(Move* move=startMoveList;move<endMoveList;move++)
                        printMove(engine->movePtr,engine->undoPtr,&engine->pos,*move,engine,move-startMoveList+1);
                    engine->movePtr=startMoveList;
                    break;
                }
                if(moveStr=="g"||moveStr=="printgame"){
                    if(UIHistory.size()==0){
                        std::cout << "nothing to print\n";
                        continue;
                    }
                    Position tempPos=UIHistory.front().pos;
                    for(int i=0;i<UIHistory.size();i++){
                        if(UIHistory[i].lastMove){
                            std::cout << algMove(engine->movePtr,engine->undoPtr,&tempPos,UIHistory[i].lastMove,engine) << " ";
                            makeMove(engine->undoPtr,&tempPos,UIHistory[i].lastMove,engine);
                            engine->undoPtr=engine->undoStack;
                        }
                    }
                    std::cout << "\n";
                    continue;
                }
                if(moveStr=="position startpos"){
                    UIHistory.clear();
                    initPosition(&engine->pos);
                    clearTT();
                    pool.clearHistories();
                    break;
                }
                if(moveStr.substr(0,13)=="position fen "){
                    UIHistory.clear();
                    initFen(&engine->pos,moveStr.substr(13).c_str());
                    clearTT();
                    pool.clearHistories();
                    break;
                }
                if(moveStr.substr(0,4)=="fen "){
                    UIHistory.clear();
                    initFen(&engine->pos,moveStr.substr(4).c_str());
                    clearTT();
                    pool.clearHistories();
                    break;
                }
                if(moveStr.substr(0,9)=="movetime "){
                    CLI_MOVETIME=atoi(moveStr.substr(9).c_str());
                    break;
                }
                if(moveStr=="stop"){
                    pool.stop();
                    continue;
                }
                if(moveStr=="trace"){
                    trace.pst[0]=0;
                    trace.pst[1]=0;
                    trace.mobility[0]=0;
                    trace.mobility[1]=0;
                    trace.pawns[0]=0;
                    trace.pawns[1]=0;
                    trace.outpost[0]=0;
                    trace.outpost[1]=0;
                    trace.bishops[0]=0;
                    trace.bishops[1]=0;
                    trace.rookFiles[0]=0;
                    trace.rookFiles[1]=0;
                    trace.kingSafety[0]=0;
                    trace.kingSafety[1]=0;
                    trace.phase=0;
                    trace.sum[0]=0;
                    trace.sum[1]=0;
                    trace.beforeAdjust=0;

                    int e=evaluate(&engine->pos);
                    uint32_t phase=trace.phase;

                    std::cout << "\nphase  " << phase << "/128\n";
                    std::cout << "pst   " << Value(lerpPhase(trace.pst[1],phase)) << "  " << Value(lerpPhase(trace.pst[0],phase)) << "\n";
                    std::cout << "mob   " << Value(lerpPhase(trace.mobility[1],phase)) << "  " << Value(lerpPhase(trace.mobility[0],phase)) << "\n";
                    std::cout << "pawn  " << Value(lerpPhase(trace.pawns[1],phase)) << "  " << Value(lerpPhase(trace.pawns[0],phase)) << "\n";
                    std::cout << "outp   " << Value(lerpPhase(trace.outpost[1],phase)) << "  " << Value(lerpPhase(trace.outpost[0],phase)) << "\n";
                    std::cout << "bish   " << Value(lerpPhase(trace.bishops[1],phase)) << "  " << Value(lerpPhase(trace.bishops[0],phase)) << "\n";
                    std::cout << "rook   " << Value(lerpPhase(trace.rookFiles[1],phase)) << "  " << Value(lerpPhase(trace.rookFiles[0],phase)) << "\n";
                    std::cout << "king   " << Value(lerpPhase(trace.kingSafety[1],phase)) << "  " << Value(lerpPhase(trace.kingSafety[0],phase)) << "\n";
                    std::cout << "sum    " << trace.sum[1] << "  " << trace.sum[0] << "\n";
                    std::cout << "diff    " << trace.beforeAdjust << "\n";
                    std::cout << "eval   " << e*(engine->pos.turn?1:-1) << "\n\n";

                    continue;
                }
                if(moveStr=="cleartt"){
                    clearTT();
                    pool.clearHistories();
                    break;
                }
                
                pool.stop();
                
                size_t moveStrLen=moveStr.size();
                size_t i=0;
                
                invalidInputs=false;

                int numMovesMade=0;
                Position currPos=engine->pos;

                while(i<moveStrLen){
                    std::string subMoveStr;

                    while(i<moveStrLen){
                        if(moveStr[i]==' '){
                            i++;
                            break;
                        }
                        subMoveStr+=moveStr[i++];
                    }

                    if(subMoveStr.empty()||subMoveStr.find('.')!=std::string::npos)
                        continue;

                    bool isValid=false;

                    Sq kingSq=counttrailingzeros(currPos.pieces[KING]&currPos.colors[currPos.turn]);
                    Bitboard checkers=sqAttackers(&currPos,kingSq,currPos.turn^1);
                    int checkDegree=popcount(checkers);
                    Move* startMoveList=engine->movePtr;
                    generateLegalMoves(engine->movePtr,&currPos,checkers,checkDegree);
                    Move* endMoveList=engine->movePtr;
                    for(Move* move=startMoveList;move<endMoveList;move++){

                        std::string uci=uciMove(&currPos,*move);
                        std::string alg=algMove(engine->movePtr,engine->undoPtr,&currPos,*move,engine);
                        
                        if(uci==subMoveStr||alg==subMoveStr||relaxedAlg(alg)==relaxedAlg(subMoveStr)){
                            UIState uiS={currPos,*move};
                            UIHistory.push_back(uiS);
                            engine->undoPtr=engine->undoStack;
                            makeMove(engine->undoPtr,&currPos,*move,engine);
                            updatePastPositionRepetition(&currPos);
                            isValid=true;
                            numMovesMade++;
                            break;
                        }
                    }
                    engine->movePtr=startMoveList;
                    
                    if(!isValid){
                        std::cout << "invalid input: " << subMoveStr<< "\n";
                        invalidInputs=true;
                        for(int i=0;i<numMovesMade;i++)
                            UIHistory.pop_back();
                        break;
                    }
                }

                if(!invalidInputs&&numMovesMade)
                    engine->pos=currPos;
            }
        }
    }
    
    return 0;
}
