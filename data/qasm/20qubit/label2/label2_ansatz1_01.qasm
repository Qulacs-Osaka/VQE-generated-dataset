OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(2.380637864101698) q[0];
rz(-1.6186412939469332) q[0];
ry(3.132062965600998) q[1];
rz(1.1012044420726248) q[1];
ry(3.1314874350071453) q[2];
rz(-1.923972255801984) q[2];
ry(1.5813122890049238) q[3];
rz(-0.6916081983883203) q[3];
ry(1.5842080524439421) q[4];
rz(3.1152930329418127) q[4];
ry(0.7472618409024725) q[5];
rz(3.1101880555068737) q[5];
ry(-0.08966358491862449) q[6];
rz(2.9471539550108927) q[6];
ry(0.3432192685555933) q[7];
rz(-0.546218971290722) q[7];
ry(0.5820014495421801) q[8];
rz(0.3550832912454691) q[8];
ry(-0.24322896023979815) q[9];
rz(-1.9896478736124001) q[9];
ry(0.007700051063140865) q[10];
rz(-0.9925511609987442) q[10];
ry(-0.04325808581670288) q[11];
rz(1.5090302539565217) q[11];
ry(1.0562414483625995) q[12];
rz(1.326618087332641) q[12];
ry(0.04627517438061357) q[13];
rz(-0.10769589214129917) q[13];
ry(3.0827453638577116) q[14];
rz(3.131116407712708) q[14];
ry(1.567971282562563) q[15];
rz(-0.12112785025983897) q[15];
ry(-1.5786859042342383) q[16];
rz(1.3991773038637503) q[16];
ry(0.0007289909227351867) q[17];
rz(1.0339209203678328) q[17];
ry(0.21425477460204953) q[18];
rz(-0.10851787258649158) q[18];
ry(-0.19231404083999118) q[19];
rz(2.2779504629066887) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.0784488691818832) q[0];
rz(2.3072019777840964) q[0];
ry(-0.04361602285612775) q[1];
rz(-0.5423696072969557) q[1];
ry(-1.5848175614908881) q[2];
rz(2.277126407455538) q[2];
ry(0.14231655340086088) q[3];
rz(-3.1175263717464268) q[3];
ry(-0.1784347401876211) q[4];
rz(-3.0384160806547635) q[4];
ry(1.5198158889457982) q[5];
rz(-1.9294327318040265) q[5];
ry(-0.09171631095856102) q[6];
rz(-2.6736410419131302) q[6];
ry(1.0871323094003174) q[7];
rz(0.14136726097853927) q[7];
ry(2.029396695730065) q[8];
rz(-1.990137172908496) q[8];
ry(1.3299984509608698) q[9];
rz(1.3542830956998184) q[9];
ry(0.013103937741623994) q[10];
rz(-1.81760148451967) q[10];
ry(3.090807387386046) q[11];
rz(0.03388492529856979) q[11];
ry(-0.397754464986789) q[12];
rz(-2.267098567603452) q[12];
ry(3.0854103821724403) q[13];
rz(-0.3220081997528882) q[13];
ry(-1.5718966842882975) q[14];
rz(3.0653422543697335) q[14];
ry(1.9250456279565498) q[15];
rz(-2.250126981076349) q[15];
ry(-1.76549144092981) q[16];
rz(1.8917104958033641) q[16];
ry(-1.5712624972004827) q[17];
rz(1.8350127023980187) q[17];
ry(-0.1927815843695772) q[18];
rz(1.3741346459983967) q[18];
ry(2.2353526542961295) q[19];
rz(-2.897625178662357) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.932630958709598) q[0];
rz(2.6623793995014076) q[0];
ry(-1.5697157003541795) q[1];
rz(1.54867327636505) q[1];
ry(-1.5884952631988885) q[2];
rz(2.896114267486505) q[2];
ry(-1.5685226464346762) q[3];
rz(-2.7262998840186383) q[3];
ry(-1.435703652862489) q[4];
rz(1.2476332460637298) q[4];
ry(-1.7399589588263131) q[5];
rz(0.8380052307126011) q[5];
ry(-2.294093987784299) q[6];
rz(-3.0314794338092867) q[6];
ry(1.3186560414668964) q[7];
rz(-0.32758449294027736) q[7];
ry(-3.105875958697346) q[8];
rz(-2.7372861241983535) q[8];
ry(-1.0697833752798995) q[9];
rz(3.026795337750306) q[9];
ry(0.17918036978197008) q[10];
rz(1.2169394487331582) q[10];
ry(-3.042858964752773) q[11];
rz(-1.5916778675884231) q[11];
ry(0.8002371749207038) q[12];
rz(-2.470962825953714) q[12];
ry(1.5468462964845007) q[13];
rz(-1.7423642032990185) q[13];
ry(1.7358834231503373) q[14];
rz(-0.5595976455984683) q[14];
ry(-0.004419111731745659) q[15];
rz(-1.9363299468760111) q[15];
ry(2.073338033969537) q[16];
rz(-1.8478890805971857) q[16];
ry(-1.5285485694442615) q[17];
rz(1.3212314074576592) q[17];
ry(1.5707396134864278) q[18];
rz(-1.5511649599572168) q[18];
ry(2.254461044848667) q[19];
rz(-0.316288550268796) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.5724146211487504) q[0];
rz(0.13669957549562284) q[0];
ry(-1.2802481073363756) q[1];
rz(1.4424636008779546) q[1];
ry(-0.07041592508586524) q[2];
rz(-1.0169981969270268) q[2];
ry(-0.25642278520423467) q[3];
rz(0.22640381699628787) q[3];
ry(-0.27481271791537215) q[4];
rz(0.520003400191114) q[4];
ry(-2.601905962795159) q[5];
rz(1.2465545929627995) q[5];
ry(0.45070711878295455) q[6];
rz(-2.8801758019108603) q[6];
ry(1.5949126864814962) q[7];
rz(-3.140710876429264) q[7];
ry(-0.026014756763810706) q[8];
rz(-3.0330259400999338) q[8];
ry(1.9371150771210148) q[9];
rz(1.4843196561107854) q[9];
ry(-3.0792682017593687) q[10];
rz(0.0625216791333356) q[10];
ry(3.072074528837996) q[11];
rz(2.663931182303903) q[11];
ry(1.56286971057909) q[12];
rz(-1.5002029761563713) q[12];
ry(-3.094583209161082) q[13];
rz(3.0485930742840215) q[13];
ry(3.054668608099538) q[14];
rz(2.5569830127618536) q[14];
ry(-0.05436882377488049) q[15];
rz(2.0255404695624497) q[15];
ry(-3.092931837362358) q[16];
rz(-0.3030212610368963) q[16];
ry(-0.044095677572580705) q[17];
rz(1.3880810977140214) q[17];
ry(2.6579377557379744) q[18];
rz(1.6176630730141115) q[18];
ry(-1.570710808334331) q[19];
rz(-0.20130897385468186) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.630104974259867) q[0];
rz(1.2514115240458352) q[0];
ry(2.0911624170594623) q[1];
rz(1.0930883628071548) q[1];
ry(-1.5733717190141374) q[2];
rz(-0.38132420375353093) q[2];
ry(0.4945404493131207) q[3];
rz(-0.13192618783717244) q[3];
ry(1.520234923057636) q[4];
rz(1.9566534391515633) q[4];
ry(-3.049056056779606) q[5];
rz(2.956405121211738) q[5];
ry(0.0029460852091771628) q[6];
rz(-2.93533594036899) q[6];
ry(-1.5216533670921482) q[7];
rz(-2.6867642559387) q[7];
ry(-0.018080877086801162) q[8];
rz(1.3523844128141886) q[8];
ry(1.294922870143049) q[9];
rz(1.1667410456882212) q[9];
ry(-0.21493368260756007) q[10];
rz(1.0021179812423284) q[10];
ry(-3.110809761827168) q[11];
rz(-2.3731198983201947) q[11];
ry(1.6710956710798786) q[12];
rz(0.6157556181497386) q[12];
ry(-1.6647457881709968) q[13];
rz(-2.4358530229020836) q[13];
ry(-0.0655257398191944) q[14];
rz(2.310349388502595) q[14];
ry(-3.0268200867351833) q[15];
rz(1.5206238361868987) q[15];
ry(1.502415136852325) q[16];
rz(-1.025267678104215) q[16];
ry(-1.578686590440734) q[17];
rz(-0.7875811934050683) q[17];
ry(1.5387323955818855) q[18];
rz(-2.373913861478286) q[18];
ry(-0.12315723564368142) q[19];
rz(-2.1735917350452816) q[19];