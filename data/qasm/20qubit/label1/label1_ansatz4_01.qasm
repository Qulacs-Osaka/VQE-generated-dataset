OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-0.582802562737246) q[0];
rz(0.619008706570396) q[0];
ry(-2.533967121590995) q[1];
rz(2.735094802282328) q[1];
ry(-0.003571233501109994) q[2];
rz(1.7733662744222163) q[2];
ry(-3.141198464182463) q[3];
rz(-2.0152674086554017) q[3];
ry(-2.5206079472092933) q[4];
rz(0.3934750445313319) q[4];
ry(-1.6256279186293847) q[5];
rz(-3.076626282617226) q[5];
ry(0.4833571326042385) q[6];
rz(-0.377015417703821) q[6];
ry(-3.0569628406362206) q[7];
rz(2.946773876754581) q[7];
ry(0.03772488363348536) q[8];
rz(2.0290696377342385) q[8];
ry(-0.08893016520397977) q[9];
rz(1.5859123694500727) q[9];
ry(-0.4398193077266104) q[10];
rz(1.5208518456625129) q[10];
ry(-2.2654996600526776) q[11];
rz(1.3973434359363557) q[11];
ry(2.2949134554486963) q[12];
rz(1.8222063006179097) q[12];
ry(-0.36017609079317126) q[13];
rz(-1.6775400047759574) q[13];
ry(-3.0636385258084324) q[14];
rz(-0.623781202283352) q[14];
ry(-3.1348556482616536) q[15];
rz(-2.8748299245849753) q[15];
ry(-2.447430442751198) q[16];
rz(2.841570372589142) q[16];
ry(0.8232306317709663) q[17];
rz(0.14456316430231012) q[17];
ry(-1.7100617272243115) q[18];
rz(0.10280976760405912) q[18];
ry(-2.9686777982211927) q[19];
rz(-2.4156402139870927) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.7623855038573435) q[0];
rz(-2.4431062960072443) q[0];
ry(1.0794139829550806) q[1];
rz(0.7944036671843369) q[1];
ry(1.5620640732195195) q[2];
rz(-2.992617359476636) q[2];
ry(-0.766361378269983) q[3];
rz(-1.5973635640989483) q[3];
ry(-3.05177463093655) q[4];
rz(-1.255690411177552) q[4];
ry(1.703343424022095) q[5];
rz(1.71045899209855) q[5];
ry(-2.7315920348040343) q[6];
rz(2.258712372507942) q[6];
ry(-1.9636296479854547) q[7];
rz(1.3888234502274839) q[7];
ry(-3.08391088154365) q[8];
rz(-0.2445181043347189) q[8];
ry(-0.02919315413214897) q[9];
rz(-0.5869811371388082) q[9];
ry(-2.6576095924628205) q[10];
rz(-0.15130457218325866) q[10];
ry(2.3064357376544837) q[11];
rz(-0.01479741145770408) q[11];
ry(2.218889159732776) q[12];
rz(-1.9880234113136277) q[12];
ry(0.4470991012379921) q[13];
rz(1.027260690248554) q[13];
ry(-3.1258032791573447) q[14];
rz(2.458422613666671) q[14];
ry(0.0052170809936864515) q[15];
rz(0.5724665402377351) q[15];
ry(2.6402292496182342) q[16];
rz(2.9431824153323847) q[16];
ry(2.2173245538959323) q[17];
rz(-0.3959562507800882) q[17];
ry(2.8816326084640864) q[18];
rz(2.7139149561163323) q[18];
ry(-0.7035112972284887) q[19];
rz(0.1306499358822301) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-2.9241214438523793) q[0];
rz(-1.5434446770777015) q[0];
ry(2.416091062546477) q[1];
rz(0.8529884095785852) q[1];
ry(1.5730874518793696) q[2];
rz(-0.005448645036260045) q[2];
ry(1.570868590748908) q[3];
rz(0.027503175175635164) q[3];
ry(-0.0324865349502117) q[4];
rz(2.472275531602238) q[4];
ry(-3.1323078701784635) q[5];
rz(1.9730002091072798) q[5];
ry(0.15987852084549026) q[6];
rz(-1.0982406667274) q[6];
ry(-1.9444890068710903) q[7];
rz(-2.868605981517188) q[7];
ry(1.9374011394838322) q[8];
rz(-2.9394303564281303) q[8];
ry(-0.6063312822217032) q[9];
rz(1.6234819273993149) q[9];
ry(-0.46661154068820054) q[10];
rz(2.605537587627581) q[10];
ry(-0.708404431614839) q[11];
rz(-2.4401632888511355) q[11];
ry(2.03982915452636) q[12];
rz(-1.1242079004726453) q[12];
ry(-1.3625221410205377) q[13];
rz(0.07097841893228464) q[13];
ry(2.1821304078183665) q[14];
rz(-0.40667569597724057) q[14];
ry(-1.149757532216868) q[15];
rz(-1.8557720895404213) q[15];
ry(-2.116456324157115) q[16];
rz(1.6672671343523824) q[16];
ry(2.7772367745062496) q[17];
rz(-0.36649325711191666) q[17];
ry(-0.25559860113421795) q[18];
rz(0.3292500281891085) q[18];
ry(0.6217855932784282) q[19];
rz(-0.2700625309441937) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.04272757325936105) q[0];
rz(0.0040843292034441385) q[0];
ry(-1.1126215663041865) q[1];
rz(1.3913441882773492) q[1];
ry(1.5620124466938725) q[2];
rz(-2.3439288681432493) q[2];
ry(1.5709093325124612) q[3];
rz(1.5505717960055094) q[3];
ry(0.0007660256665087951) q[4];
rz(-2.3614279162302676) q[4];
ry(3.1404642368128535) q[5];
rz(1.807466622824702) q[5];
ry(-0.2843598486954583) q[6];
rz(-3.141497497049318) q[6];
ry(1.5911327679208742) q[7];
rz(-1.567995454535656) q[7];
ry(-0.01273682805350539) q[8];
rz(-1.7072511262025243) q[8];
ry(0.08089247389241461) q[9];
rz(-0.05628919116401713) q[9];
ry(3.095246567043699) q[10];
rz(-0.014063685122341482) q[10];
ry(-0.022283385726400134) q[11];
rz(-0.5223751777896692) q[11];
ry(-2.7850628897107224) q[12];
rz(-1.4648868633520626) q[12];
ry(-2.9241872502867476) q[13];
rz(-2.249225260506085) q[13];
ry(3.126917149594762) q[14];
rz(1.1812536595531284) q[14];
ry(3.1337277658235827) q[15];
rz(-0.2838283491628024) q[15];
ry(1.5472816695680558) q[16];
rz(-1.8437939525021623) q[16];
ry(-3.106409449071025) q[17];
rz(1.8455434521274021) q[17];
ry(2.4311051374151584) q[18];
rz(-1.7907102914793536) q[18];
ry(0.2027556115352234) q[19];
rz(0.7020943796893535) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.19187184252674735) q[0];
rz(1.3991559370712041) q[0];
ry(-1.5776625389856185) q[1];
rz(-0.13863857111021183) q[1];
ry(1.5738074885549433) q[2];
rz(-0.1345054363548241) q[2];
ry(2.2130150454620905) q[3];
rz(-1.7427890250832325) q[3];
ry(-2.0160497619887776) q[4];
rz(-1.7398047477109557) q[4];
ry(1.82182930902702) q[5];
rz(-1.7463327965771405) q[5];
ry(1.3134008583599108) q[6];
rz(-1.7533971260257655) q[6];
ry(2.442169515789083) q[7];
rz(1.4067613085030413) q[7];
ry(-2.846245744234447) q[8];
rz(-1.7058836383332228) q[8];
ry(-2.3208836046603047) q[9];
rz(1.3829513334852654) q[9];
ry(-2.621207435855875) q[10];
rz(-1.758317150400475) q[10];
ry(-2.967146551726453) q[11];
rz(-2.2243683323475625) q[11];
ry(-1.6751260084857642) q[12];
rz(-0.12633338178194717) q[12];
ry(-1.5121570965644897) q[13];
rz(-0.12494124130637817) q[13];
ry(2.569626091723826) q[14];
rz(1.4510810552810038) q[14];
ry(2.3652785018375653) q[15];
rz(-1.6935169665631098) q[15];
ry(-1.6690435040805927) q[16];
rz(0.21949256056230657) q[16];
ry(0.03566457181722793) q[17];
rz(-1.720030710237301) q[17];
ry(0.012089395205259201) q[18];
rz(-1.4098677875885253) q[18];
ry(0.005668570609912129) q[19];
rz(-0.604022763911563) q[19];