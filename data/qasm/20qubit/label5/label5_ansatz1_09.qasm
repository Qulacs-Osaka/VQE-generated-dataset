OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-1.8964531978481798) q[0];
rz(-2.3925593092499806) q[0];
ry(-3.1307001249377038) q[1];
rz(-1.2800087724850486) q[1];
ry(-2.2121467449680625) q[2];
rz(-1.2034145152452274) q[2];
ry(-2.093449624464732) q[3];
rz(-1.5978335148862859) q[3];
ry(-3.1324360032926144) q[4];
rz(-1.647812140068864) q[4];
ry(1.668724135650013) q[5];
rz(-1.770984088566416) q[5];
ry(1.6515925523715858) q[6];
rz(2.89330021681774) q[6];
ry(-0.04259972123399662) q[7];
rz(-1.0284473819939635) q[7];
ry(0.15627935566226053) q[8];
rz(-3.0730764377308555) q[8];
ry(-0.638546752074328) q[9];
rz(0.00473114284076015) q[9];
ry(1.565991802424425) q[10];
rz(-0.7693054354710472) q[10];
ry(1.5649204418710347) q[11];
rz(0.19300322467595973) q[11];
ry(-0.0009177276701402586) q[12];
rz(-1.5411514350257585) q[12];
ry(-1.5673330120859852) q[13];
rz(0.6355808757752115) q[13];
ry(-1.573572539939371) q[14];
rz(1.6213629020347224) q[14];
ry(-1.5099164903241977) q[15];
rz(-0.6006612885304792) q[15];
ry(-0.2330038857163482) q[16];
rz(1.414777419646776) q[16];
ry(-2.5118808812168543) q[17];
rz(-2.4115061112707434) q[17];
ry(-2.7504055880235025) q[18];
rz(-1.6672373960683065) q[18];
ry(-0.13330866420706133) q[19];
rz(-0.7726417525194571) q[19];
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
ry(-1.637041308603468) q[0];
rz(0.5194442749829093) q[0];
ry(1.605740182535735) q[1];
rz(2.9546298924344394) q[1];
ry(-0.6425217802800353) q[2];
rz(-0.42894288980564266) q[2];
ry(-1.5560398828213973) q[3];
rz(0.5155449915207065) q[3];
ry(1.692043875701991) q[4];
rz(-1.2139604762543916) q[4];
ry(-2.009374467024008) q[5];
rz(-2.53152869106962) q[5];
ry(-0.6841108838053368) q[6];
rz(1.1588545239727575) q[6];
ry(1.612681897946142) q[7];
rz(-0.7779135825304628) q[7];
ry(2.622201988022828) q[8];
rz(2.0299376626177006) q[8];
ry(0.9289106787349019) q[9];
rz(2.9435618834445245) q[9];
ry(0.06618161151592959) q[10];
rz(2.469590604548634) q[10];
ry(-1.6708604492506) q[11];
rz(-1.9549829755107366) q[11];
ry(-1.5103138806147118) q[12];
rz(-0.13345068916000627) q[12];
ry(3.1394902151694155) q[13];
rz(3.099208629704235) q[13];
ry(1.6367114146081805) q[14];
rz(1.5731383821742024) q[14];
ry(-0.08999373371487414) q[15];
rz(-2.5532745756380595) q[15];
ry(3.063392042406326) q[16];
rz(3.037153187222365) q[16];
ry(1.6455980346209993) q[17];
rz(1.8400402051414249) q[17];
ry(-1.9911839214646814) q[18];
rz(-1.6822556953424284) q[18];
ry(1.1543273729597603) q[19];
rz(2.530546180958449) q[19];
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
ry(-2.701344317013537) q[0];
rz(-2.621915186955658) q[0];
ry(0.03490584467008695) q[1];
rz(1.967094236404649) q[1];
ry(-1.5774309172532344) q[2];
rz(2.5934823727697838) q[2];
ry(0.5131095296739767) q[3];
rz(-3.1384255922628173) q[3];
ry(1.7010742285791902) q[4];
rz(1.3104186906345516) q[4];
ry(-0.9120896589761047) q[5];
rz(0.6957867211412028) q[5];
ry(0.45132864432644837) q[6];
rz(-0.37793231107065356) q[6];
ry(3.135016003869859) q[7];
rz(0.8724209206745055) q[7];
ry(0.0004855542930375378) q[8];
rz(-1.2058814948335577) q[8];
ry(-1.0149913817413836) q[9];
rz(-2.605624927480771) q[9];
ry(0.031323431463691875) q[10];
rz(-1.2630203470388057) q[10];
ry(1.57613816417214) q[11];
rz(-0.799066347198876) q[11];
ry(-0.002850766449811637) q[12];
rz(2.801454985072343) q[12];
ry(-2.529545202331007) q[13];
rz(-0.015706353840638343) q[13];
ry(-1.57551910166993) q[14];
rz(0.6062959392224148) q[14];
ry(-0.657949564086047) q[15];
rz(-2.200806394391558) q[15];
ry(1.4881727377563216) q[16];
rz(2.8713392106141824) q[16];
ry(0.41507460155968484) q[17];
rz(2.034404288586824) q[17];
ry(0.010687078463343597) q[18];
rz(-1.3722639745014718) q[18];
ry(-2.4259841587901834) q[19];
rz(1.110083804905571) q[19];
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
ry(1.3280649280617496) q[0];
rz(-2.071032882621961) q[0];
ry(-1.5411650745774468) q[1];
rz(1.6626147779339293) q[1];
ry(-1.669653362357246) q[2];
rz(2.8892294135595313) q[2];
ry(1.8786754302849353) q[3];
rz(0.0025243745108429754) q[3];
ry(0.05714281299535609) q[4];
rz(2.8669602781126815) q[4];
ry(0.004489157961448775) q[5];
rz(-2.094514298737841) q[5];
ry(3.0037548462369923) q[6];
rz(0.575808835549141) q[6];
ry(0.9531627696788152) q[7];
rz(-1.0206782623643074) q[7];
ry(2.2974372837672035) q[8];
rz(3.122121444764873) q[8];
ry(0.08705431554337206) q[9];
rz(-0.43662585085453465) q[9];
ry(0.7777687411099681) q[10];
rz(1.5765619827543897) q[10];
ry(-0.2332986431146229) q[11];
rz(0.7962250797670345) q[11];
ry(1.1193844729239837) q[12];
rz(-2.62446900300269) q[12];
ry(-2.342008588714334) q[13];
rz(-0.9878400939280425) q[13];
ry(1.5338685853246403) q[14];
rz(-1.6472489056351076) q[14];
ry(1.6281484585538064) q[15];
rz(-1.9799066206942335) q[15];
ry(1.6137611118539672) q[16];
rz(0.09635283018687357) q[16];
ry(1.1873311438725098) q[17];
rz(-0.09747861371706978) q[17];
ry(-0.5316588453357122) q[18];
rz(1.9016255943211322) q[18];
ry(-1.0973309717898103) q[19];
rz(0.6326739927892185) q[19];
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
ry(1.7250625979229) q[0];
rz(2.8771001526265354) q[0];
ry(2.833610333722504) q[1];
rz(-0.8845463612279819) q[1];
ry(1.7506968215147527) q[2];
rz(1.165559430345034) q[2];
ry(-2.0009314127896896) q[3];
rz(0.002202418214983126) q[3];
ry(0.0013753677517103213) q[4];
rz(1.9987840172728015) q[4];
ry(-3.0630811596021563) q[5];
rz(-3.133157006229551) q[5];
ry(1.3958541797819528) q[6];
rz(1.3574813837833795) q[6];
ry(0.0021549085183004283) q[7];
rz(1.08094212816354) q[7];
ry(-3.140110574952731) q[8];
rz(-0.9900528890870008) q[8];
ry(1.4824259831146316) q[9];
rz(3.0155719185425913) q[9];
ry(1.5736900856136744) q[10];
rz(-1.5798718599716575) q[10];
ry(-0.019431291960066006) q[11];
rz(-1.5817729467000332) q[11];
ry(3.135867102178356) q[12];
rz(-2.6311292825319725) q[12];
ry(-0.00018186700702041544) q[13];
rz(-0.35223938621791906) q[13];
ry(-1.5487809433074187) q[14];
rz(1.5666491270715168) q[14];
ry(0.9774569933493842) q[15];
rz(0.9139401978427656) q[15];
ry(2.8319236018512797) q[16];
rz(-1.2878079182069175) q[16];
ry(0.8693029648947741) q[17];
rz(0.5722571175639191) q[17];
ry(-2.9818738339221986) q[18];
rz(1.062205214075906) q[18];
ry(3.058646674426894) q[19];
rz(0.6788237115928215) q[19];
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
ry(2.9883114738095555) q[0];
rz(-0.14599405886744776) q[0];
ry(-3.1139187515925126) q[1];
rz(-0.9099544595115825) q[1];
ry(2.844446979291173) q[2];
rz(-2.0776924186425303) q[2];
ry(-0.7078308032924925) q[3];
rz(-3.1415696385417258) q[3];
ry(1.5874506543955846) q[4];
rz(-1.2597245793881733) q[4];
ry(-0.009278126172886158) q[5];
rz(-0.8327525028484528) q[5];
ry(-1.5805837376890786) q[6];
rz(1.6719694181641913) q[6];
ry(2.3729801137556494) q[7];
rz(-1.2768928872669432) q[7];
ry(1.3573550511404167) q[8];
rz(-1.7726026342027164) q[8];
ry(-1.6068925122833106) q[9];
rz(-1.4084592178963902) q[9];
ry(1.6007228415202137) q[10];
rz(0.47297958779190996) q[10];
ry(-1.5687912511222848) q[11];
rz(1.5343047368090068) q[11];
ry(-1.125749797520415) q[12];
rz(3.0844555229879047) q[12];
ry(3.1320809129600367) q[13];
rz(2.4303641994762257) q[13];
ry(-1.567871603624) q[14];
rz(-2.2683819248536112) q[14];
ry(-0.014062658257316231) q[15];
rz(2.900727129819308) q[15];
ry(-3.100906220465996) q[16];
rz(-0.555273023316998) q[16];
ry(0.055877479917394446) q[17];
rz(-3.0629499521279433) q[17];
ry(0.007462670125929052) q[18];
rz(-0.12446291060556992) q[18];
ry(2.1570491451624765) q[19];
rz(-1.781380514198417) q[19];
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
ry(-1.1554785292005774) q[0];
rz(-3.0203594513066934) q[0];
ry(0.2967734924328154) q[1];
rz(0.4001452428303982) q[1];
ry(-0.040935979596351924) q[2];
rz(0.2552649627284547) q[2];
ry(-2.6913673981983437) q[3];
rz(-0.9959140620521945) q[3];
ry(0.005958541317877319) q[4];
rz(0.4870786136614233) q[4];
ry(0.0008703660113136279) q[5];
rz(1.069201440988913) q[5];
ry(-1.5804091923189576) q[6];
rz(1.1910166575429162) q[6];
ry(-1.5585021022105743) q[7];
rz(1.5812787479556798) q[7];
ry(0.0013602155633787303) q[8];
rz(2.7295973385387993) q[8];
ry(-0.01917931276260809) q[9];
rz(0.12408293106145883) q[9];
ry(-0.17872630479340668) q[10];
rz(3.0991406835429585) q[10];
ry(1.8570669884639948) q[11];
rz(-1.6197495873776016) q[11];
ry(-1.4471302823630308) q[12];
rz(-2.3090591351323595) q[12];
ry(0.010363035827164245) q[13];
rz(0.10176475568485992) q[13];
ry(-0.1150001969024608) q[14];
rz(-2.3745272673520663) q[14];
ry(-1.5797750071257477) q[15];
rz(1.568759667998081) q[15];
ry(0.009919431425790215) q[16];
rz(-1.2750587877222401) q[16];
ry(2.235866956664826) q[17];
rz(1.823661594032477) q[17];
ry(1.359300905531179) q[18];
rz(-2.404628458553011) q[18];
ry(-0.5824967213509175) q[19];
rz(-1.2879177988347472) q[19];
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
ry(1.2398227120834935) q[0];
rz(0.5685519398585455) q[0];
ry(-0.004477822650906127) q[1];
rz(-3.1034579336374475) q[1];
ry(-1.8371302611846814) q[2];
rz(-1.390168804640742) q[2];
ry(-0.008811744473247174) q[3];
rz(0.9820998396940864) q[3];
ry(0.1365100669636722) q[4];
rz(-2.3257041764341992) q[4];
ry(-1.478962609980166) q[5];
rz(-0.0007193426830642836) q[5];
ry(1.5801296534960967) q[6];
rz(-1.5659271533503656) q[6];
ry(-1.5745272533607653) q[7];
rz(2.3956992080011297) q[7];
ry(0.0002469166779617282) q[8];
rz(2.5213949482172353) q[8];
ry(-3.042230113265921) q[9];
rz(0.08366143940387613) q[9];
ry(3.0275434099815834) q[10];
rz(1.9987079004736108) q[10];
ry(0.004571367045361541) q[11];
rz(1.523844004662363) q[11];
ry(-3.13844643726653) q[12];
rz(-2.5321929570453507) q[12];
ry(1.0745875069519786) q[13];
rz(-2.665248244242144) q[13];
ry(2.7846336880243387) q[14];
rz(-0.7730681207887065) q[14];
ry(2.3258796737856837) q[15];
rz(-1.179908597841246) q[15];
ry(0.21956824540554054) q[16];
rz(0.42650534436653637) q[16];
ry(-1.861023575465413) q[17];
rz(0.2124382479886967) q[17];
ry(3.045632431437441) q[18];
rz(0.5985983949108289) q[18];
ry(0.14694313527714087) q[19];
rz(-0.20624122613217694) q[19];
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
ry(-0.32098006569170323) q[0];
rz(-2.001171204287333) q[0];
ry(0.015217099953604318) q[1];
rz(1.1022349807543137) q[1];
ry(-2.9429926840087632) q[2];
rz(-1.058831775584136) q[2];
ry(-2.613948927628401) q[3];
rz(-2.322359442537896) q[3];
ry(1.5116428325396178) q[4];
rz(-0.008356945970890097) q[4];
ry(2.2667397773982687) q[5];
rz(0.00022208930671732845) q[5];
ry(-1.9928965397538754) q[6];
rz(-3.136099658716282) q[6];
ry(-1.5639982838522721) q[7];
rz(1.5837664231758586) q[7];
ry(1.5384824290976877) q[8];
rz(0.007317786685998584) q[8];
ry(3.1402506615073493) q[9];
rz(2.9350647628369044) q[9];
ry(1.44347659628929) q[10];
rz(-2.8929291872696044) q[10];
ry(-0.28895063356776973) q[11];
rz(1.6529949940936686) q[11];
ry(-3.1413867931408026) q[12];
rz(-0.18854935801694148) q[12];
ry(-0.00921944441453526) q[13];
rz(1.0172214977435754) q[13];
ry(3.1379244887736957) q[14];
rz(2.109391869926879) q[14];
ry(0.013488325895601783) q[15];
rz(-0.189544487474083) q[15];
ry(-2.4055369944956633) q[16];
rz(-3.1381094185257052) q[16];
ry(3.129186088285208) q[17];
rz(-2.9334350808769014) q[17];
ry(-1.4501766987388924) q[18];
rz(-0.7321063229695133) q[18];
ry(-2.3175946631802984) q[19];
rz(0.2238473682021923) q[19];
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
ry(1.6093024809131125) q[0];
rz(-2.0545977794379606) q[0];
ry(1.9432660512493225) q[1];
rz(3.126182883701046) q[1];
ry(-2.063324735084903) q[2];
rz(0.36216780660448966) q[2];
ry(-3.14143997794378) q[3];
rz(-2.3127392672720273) q[3];
ry(0.8011910145442567) q[4];
rz(0.049136367976626745) q[4];
ry(2.4506564063334886) q[5];
rz(3.1400444973766906) q[5];
ry(-1.7680153800994114) q[6];
rz(0.00190404912078113) q[6];
ry(-1.4497469560351828) q[7];
rz(3.140597280982812) q[7];
ry(-2.0987201252264747) q[8];
rz(-0.0030726790309421105) q[8];
ry(1.545150666144579) q[9];
rz(-3.1414218965779646) q[9];
ry(-1.6112174306101195) q[10];
rz(-2.663353237087836) q[10];
ry(0.39991861805438766) q[11];
rz(0.5167893891526667) q[11];
ry(0.4866567441752929) q[12];
rz(0.1726529204833387) q[12];
ry(2.335458633940556) q[13];
rz(1.1512474265246482) q[13];
ry(-1.6969920558494733) q[14];
rz(-2.4855075812439646) q[14];
ry(-3.139586989516007) q[15];
rz(1.7934087937105572) q[15];
ry(1.5360230941449033) q[16];
rz(2.5672623656326534) q[16];
ry(-2.630990971790004) q[17];
rz(0.012494773029937178) q[17];
ry(0.03746101661389378) q[18];
rz(0.7604568450414089) q[18];
ry(1.9850224575342086) q[19];
rz(-2.8601502452320045) q[19];
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
ry(2.0323664974923488) q[0];
rz(3.115656818714382) q[0];
ry(1.711373717796529) q[1];
rz(0.000513880921193453) q[1];
ry(-3.0357452706008456) q[2];
rz(0.0905627681289624) q[2];
ry(-1.8739331439228781) q[3];
rz(-0.06932911273793517) q[3];
ry(0.030119975478593398) q[4];
rz(1.9251146027315835) q[4];
ry(2.3082092164591104) q[5];
rz(3.1412400676432446) q[5];
ry(1.370724831134613) q[6];
rz(0.00025173627410022024) q[6];
ry(-1.3537760914387205) q[7];
rz(-3.14087155780584) q[7];
ry(-0.6307072454937503) q[8];
rz(0.004962778456641705) q[8];
ry(-1.8944443662168875) q[9];
rz(0.04026146324195334) q[9];
ry(1.1313739898578956) q[10];
rz(2.310951006889451) q[10];
ry(-3.1290798153298542) q[11];
rz(-2.424643843170452) q[11];
ry(0.39970617633681904) q[12];
rz(2.9510550246566862) q[12];
ry(-1.7760896449788641) q[13];
rz(3.1406711374863128) q[13];
ry(1.8897236886032438) q[14];
rz(-0.0028870945924230895) q[14];
ry(-0.8707131737210467) q[15];
rz(3.130225810373341) q[15];
ry(2.9891922651557374) q[16];
rz(-0.6471532428870734) q[16];
ry(-1.7207251947374242) q[17];
rz(-0.005643383920638456) q[17];
ry(1.0537982747359944) q[18];
rz(0.2305184804613276) q[18];
ry(-3.0956647416329504) q[19];
rz(-2.61703891080358) q[19];
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
ry(3.1343944585525376) q[0];
rz(-1.9858437330382746) q[0];
ry(0.7967146643782209) q[1];
rz(-1.5662022981149573) q[1];
ry(-1.7790565070687947) q[2];
rz(1.5917724905945088) q[2];
ry(0.0405085573742419) q[3];
rz(1.6376738685396302) q[3];
ry(-0.003879433563678702) q[4];
rz(-0.40088357665516566) q[4];
ry(-1.7473625123542624) q[5];
rz(-1.5705626318483747) q[5];
ry(-1.470841725777873) q[6];
rz(1.5709148413759884) q[6];
ry(-2.0205525078407702) q[7];
rz(1.571453722996007) q[7];
ry(-2.3579105877424937) q[8];
rz(1.5478494961651643) q[8];
ry(-0.0433774706352974) q[9];
rz(1.5278215330710954) q[9];
ry(3.130601163545868) q[10];
rz(0.06451530070941745) q[10];
ry(-3.112706422078538) q[11];
rz(1.7781986228700717) q[11];
ry(0.6642397439877172) q[12];
rz(1.5605581650040197) q[12];
ry(1.6397372145157745) q[13];
rz(1.5689497626055822) q[13];
ry(-1.3413742927669405) q[14];
rz(-1.5737983676824552) q[14];
ry(-2.782916944178307) q[15];
rz(-1.5746756316545705) q[15];
ry(2.090918239105159) q[16];
rz(1.5577963183955983) q[16];
ry(2.222527780155409) q[17];
rz(-1.5779939586591025) q[17];
ry(-0.3902638674048502) q[18];
rz(1.4324996826302898) q[18];
ry(2.6603644396554316) q[19];
rz(-1.4112475532429078) q[19];
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
ry(-0.12296117479416324) q[0];
rz(-1.506174256137034) q[0];
ry(-1.5712997725821225) q[1];
rz(-0.06081752000004552) q[1];
ry(-1.5713754903476458) q[2];
rz(-1.92612480357458) q[2];
ry(-1.5709267162191791) q[3];
rz(-1.0430733085810422) q[3];
ry(1.576169868005614) q[4];
rz(2.7498349360515872) q[4];
ry(-1.5710617866896683) q[5];
rz(1.1006940445070894) q[5];
ry(-1.5698047620928097) q[6];
rz(-1.9419856875577446) q[6];
ry(-1.5703127274145796) q[7];
rz(-0.43657403279466855) q[7];
ry(-1.579109109328643) q[8];
rz(-1.6399776935471984) q[8];
ry(-1.5706956871113789) q[9];
rz(0.6212703921334723) q[9];
ry(0.5983392378417777) q[10];
rz(1.9549124631142025) q[10];
ry(1.5667707902418668) q[11];
rz(2.8105535724293937) q[11];
ry(1.5779268316781727) q[12];
rz(1.335498745809449) q[12];
ry(1.5705166121402632) q[13];
rz(-1.8552052706045858) q[13];
ry(1.5792104864966916) q[14];
rz(1.512466494370922) q[14];
ry(1.5687981481784874) q[15];
rz(1.4481652186620666) q[15];
ry(1.5937797064269024) q[16];
rz(2.5528489007320205) q[16];
ry(1.5735204684302235) q[17];
rz(-3.1119866076750866) q[17];
ry(-1.6566336225271145) q[18];
rz(-3.084722216198638) q[18];
ry(-1.5824249851455559) q[19];
rz(0.7644227327512412) q[19];