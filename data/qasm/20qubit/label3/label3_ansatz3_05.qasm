OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-3.0562524397741626) q[0];
rz(-2.191886148437952) q[0];
ry(1.150506844026115) q[1];
rz(1.2965157895320782) q[1];
ry(3.140177398252886) q[2];
rz(2.995667839328446) q[2];
ry(2.098273156876817) q[3];
rz(2.6292100686552256) q[3];
ry(-1.4889906521391936) q[4];
rz(0.5793162505984618) q[4];
ry(3.141038923113396) q[5];
rz(1.5812838710077326) q[5];
ry(-0.48559452769961814) q[6];
rz(1.1612117810936682) q[6];
ry(-1.120426678999669) q[7];
rz(-1.1167609003263703) q[7];
ry(-2.0343982579260813) q[8];
rz(0.07669254943838619) q[8];
ry(-0.09089798000575033) q[9];
rz(2.915080650544792) q[9];
ry(-0.033138780199412396) q[10];
rz(-1.118351383340766) q[10];
ry(-2.64763644525045) q[11];
rz(0.2678204862615372) q[11];
ry(-3.1316834819745947) q[12];
rz(1.057438825735343) q[12];
ry(-0.24555949462503343) q[13];
rz(0.10061392417984827) q[13];
ry(-0.331126525512321) q[14];
rz(2.9841574436132317) q[14];
ry(-2.9275458425351903) q[15];
rz(-0.35262517064256743) q[15];
ry(-3.021488926612292) q[16];
rz(-0.1765019575280144) q[16];
ry(1.6075448065713414) q[17];
rz(-2.9254008131124154) q[17];
ry(2.7806484943378167) q[18];
rz(-1.476549246978462) q[18];
ry(-0.3762052391813937) q[19];
rz(2.3831250675227333) q[19];
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
ry(-1.900757226699789) q[0];
rz(-2.682761229479733) q[0];
ry(-1.8515648167343448) q[1];
rz(-2.1960821550744587) q[1];
ry(1.5706431788196333) q[2];
rz(-1.653195106734475) q[2];
ry(1.202455023398118) q[3];
rz(-0.13329739782511882) q[3];
ry(0.008242936536921874) q[4];
rz(-0.7033185178522237) q[4];
ry(-1.5855862530615727) q[5];
rz(2.4103011268481067) q[5];
ry(-0.0014467868273069229) q[6];
rz(1.8125734381000038) q[6];
ry(0.1796078128034026) q[7];
rz(0.23822590007809344) q[7];
ry(1.784749908797977) q[8];
rz(-0.21415822063344425) q[8];
ry(-0.004836980007683975) q[9];
rz(0.043714401422563214) q[9];
ry(0.6966905507383254) q[10];
rz(-0.2641630395554264) q[10];
ry(-2.0399329132421036) q[11];
rz(-0.112662146829119) q[11];
ry(3.1172866731460402) q[12];
rz(-1.8721756147568511) q[12];
ry(-0.23578610227957864) q[13];
rz(-3.0320536776117035) q[13];
ry(-2.8346667901252234) q[14];
rz(1.833131448776502) q[14];
ry(-0.8372593555784089) q[15];
rz(0.3597285660041431) q[15];
ry(-0.15837465759875824) q[16];
rz(3.1014219347900354) q[16];
ry(1.4853144730167531) q[17];
rz(0.9631382333119706) q[17];
ry(1.5093223125473265) q[18];
rz(-1.7613890007460853) q[18];
ry(-1.6649428132580972) q[19];
rz(-1.1855771116462965) q[19];
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
ry(1.5705857466832907) q[0];
rz(-0.001687109808496956) q[0];
ry(2.6839126883027093) q[1];
rz(-1.141523709960973) q[1];
ry(-0.001170226686440239) q[2];
rz(-3.0588646562455377) q[2];
ry(-0.4824660244525531) q[3];
rz(1.8911687502042813) q[3];
ry(-1.3194972368184301) q[4];
rz(1.8879356604201096) q[4];
ry(-3.1374447093559796) q[5];
rz(1.7521745844217325) q[5];
ry(0.07788270978731608) q[6];
rz(-2.798014094994013) q[6];
ry(1.5729111414744725) q[7];
rz(-3.1273271868611148) q[7];
ry(0.3344353398008448) q[8];
rz(-2.485882666186985) q[8];
ry(-0.8018443309955003) q[9];
rz(-1.3550358874671191) q[9];
ry(-0.021026863905747817) q[10];
rz(-0.8784127440310012) q[10];
ry(0.018389476262082753) q[11];
rz(1.7992124509655891) q[11];
ry(-3.134154074180948) q[12];
rz(2.029350572545403) q[12];
ry(-0.06382394564885807) q[13];
rz(2.3989996780862155) q[13];
ry(1.627994537751288) q[14];
rz(1.3956297850790924) q[14];
ry(-2.5099766659552443) q[15];
rz(3.1332173550387625) q[15];
ry(1.529691136613204) q[16];
rz(-1.0594228885189727) q[16];
ry(-2.1405055059833415) q[17];
rz(-2.539219224508595) q[17];
ry(-1.773099837740169) q[18];
rz(-3.0241777846171956) q[18];
ry(-2.840002274455287) q[19];
rz(-1.9167839739217714) q[19];
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
ry(1.6042921099691505) q[0];
rz(1.5578786873909811) q[0];
ry(1.571802285166946) q[1];
rz(-1.5700348399314217) q[1];
ry(1.5722863491849886) q[2];
rz(-1.53807149277843) q[2];
ry(1.7409466228510917) q[3];
rz(-2.736072071052835) q[3];
ry(3.10864079852393) q[4];
rz(2.269327116674554) q[4];
ry(0.013392193353203297) q[5];
rz(-2.7400283657505273) q[5];
ry(0.01188340977276603) q[6];
rz(-3.032644147320711) q[6];
ry(-0.22923637121399576) q[7];
rz(1.5480404951961715) q[7];
ry(-3.085500024398409) q[8];
rz(-2.231351859671529) q[8];
ry(-2.6207698226124183) q[9];
rz(2.9461165786456815) q[9];
ry(-1.3212101204611741) q[10];
rz(-2.9772669065382598) q[10];
ry(1.6920704563803142) q[11];
rz(-1.1997416298307604) q[11];
ry(3.125619040325457) q[12];
rz(-1.721025899601633) q[12];
ry(3.140037275309959) q[13];
rz(1.6218724276766454) q[13];
ry(1.5819365861717731) q[14];
rz(3.087265472785494) q[14];
ry(-1.5840710198892218) q[15];
rz(-0.8117097203663004) q[15];
ry(-3.0071211712962334) q[16];
rz(-0.10383891869390728) q[16];
ry(0.030895221317836834) q[17];
rz(2.107410623905898) q[17];
ry(-2.4953076012988777) q[18];
rz(-1.7872797391661017) q[18];
ry(-2.790941163184162) q[19];
rz(2.374967135124522) q[19];
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
ry(3.0846508805042143) q[0];
rz(-1.5842908891479937) q[0];
ry(-1.5671648728353347) q[1];
rz(1.3866550353508782) q[1];
ry(1.849829637729559) q[2];
rz(1.5862241355775994) q[2];
ry(-3.0733909884472252) q[3];
rz(1.0366545422703144) q[3];
ry(-1.5716747975646819) q[4];
rz(1.5727085799283866) q[4];
ry(-3.141442213240431) q[5];
rz(1.320285791028999) q[5];
ry(-1.5533786531744855) q[6];
rz(2.5171508134690392) q[6];
ry(-0.7591872475414414) q[7];
rz(0.3012707620358791) q[7];
ry(0.25862755218464883) q[8];
rz(0.1926824657796919) q[8];
ry(-2.8580565267489737) q[9];
rz(-0.8029475158002324) q[9];
ry(-3.131964169901332) q[10];
rz(-0.44717809793638974) q[10];
ry(-0.01600149629814762) q[11];
rz(-0.3573945237455801) q[11];
ry(3.0483129533603344) q[12];
rz(-2.1724029608272772) q[12];
ry(-0.10615815926024563) q[13];
rz(-1.3100708276934308) q[13];
ry(-1.6093779695916135) q[14];
rz(1.4528189062921069) q[14];
ry(-0.21963971023287276) q[15];
rz(-1.0477515136139492) q[15];
ry(-1.586108808492842) q[16];
rz(-1.8606101414032041) q[16];
ry(-0.0017959829179471682) q[17];
rz(0.7024418896605389) q[17];
ry(-1.3719920755895332) q[18];
rz(-2.5582646570313763) q[18];
ry(-3.1301317492475076) q[19];
rz(1.7155795452107485) q[19];
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
ry(1.5694082215913303) q[0];
rz(-2.3750402107633533) q[0];
ry(-2.113719191745818) q[1];
rz(0.06378925839075812) q[1];
ry(-1.2883167294656233) q[2];
rz(-3.0806632215449135) q[2];
ry(-0.000696371437644038) q[3];
rz(-2.5715758882858677) q[3];
ry(-0.892635652179653) q[4];
rz(3.141230427012014) q[4];
ry(0.009669306643410466) q[5];
rz(-2.2899159589076774) q[5];
ry(3.1408669193307626) q[6];
rz(0.9748828405761363) q[6];
ry(0.028440346029277086) q[7];
rz(2.7930166177318805) q[7];
ry(0.1792385767540561) q[8];
rz(0.542497283124833) q[8];
ry(-2.7237595702442627) q[9];
rz(-3.1053126612023694) q[9];
ry(-1.518055591220707) q[10];
rz(-0.12986500455644065) q[10];
ry(-1.5645463809622704) q[11];
rz(-0.16894245807250158) q[11];
ry(-1.6432558327121072) q[12];
rz(0.8513954308724947) q[12];
ry(1.7388830136616473) q[13];
rz(-3.0173469715479864) q[13];
ry(-1.759102768106957) q[14];
rz(-3.130191116583627) q[14];
ry(-1.7043463274307484) q[15];
rz(-1.2647235578192753) q[15];
ry(0.9743810877543087) q[16];
rz(-1.4946200372773142) q[16];
ry(1.510545990090284) q[17];
rz(-1.4886763649633998) q[17];
ry(1.4234573240445414) q[18];
rz(-0.09809572087345006) q[18];
ry(2.5036465105522985) q[19];
rz(-2.9490721195082332) q[19];
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
ry(1.5704125537360216) q[0];
rz(2.9181161187305253) q[0];
ry(1.4577450518084172) q[1];
rz(1.7208031544462947) q[1];
ry(-1.567596165619861) q[2];
rz(1.5702151964176485) q[2];
ry(0.2569124697596194) q[3];
rz(-2.1822655117467553) q[3];
ry(1.558824942171972) q[4];
rz(0.037684541622095225) q[4];
ry(-0.000780555555874076) q[5];
rz(-1.6879924298533275) q[5];
ry(1.5334136274702765) q[6];
rz(-0.04790187199430384) q[6];
ry(1.653998880287778) q[7];
rz(-0.9725144298166858) q[7];
ry(3.1087478812967038) q[8];
rz(-0.45257771037491773) q[8];
ry(3.1330597089196184) q[9];
rz(0.434535711088144) q[9];
ry(0.3589015072039956) q[10];
rz(2.954467889413114) q[10];
ry(2.8845298393472025) q[11];
rz(2.632332290774257) q[11];
ry(-3.1346229002801596) q[12];
rz(-2.3033813394572515) q[12];
ry(-0.040040797621595785) q[13];
rz(3.004172419323825) q[13];
ry(0.4454046043166419) q[14];
rz(-2.0943190448433495) q[14];
ry(-2.8527029281623886) q[15];
rz(-0.04507414447956304) q[15];
ry(-0.4103655109446649) q[16];
rz(-2.7395314810424654) q[16];
ry(3.0675086582014837) q[17];
rz(-3.026224405946591) q[17];
ry(-1.6463362583685095) q[18];
rz(0.3273457052886011) q[18];
ry(1.3921306896373906) q[19];
rz(-1.4538145356228418) q[19];
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
ry(0.014321126746325973) q[0];
rz(3.0251643601971345) q[0];
ry(-1.571728989617822) q[1];
rz(-1.339196595893539) q[1];
ry(-1.5551680959351932) q[2];
rz(1.010483013693622) q[2];
ry(1.5646283750124272) q[3];
rz(0.003260895998501212) q[3];
ry(-0.0008814516137032421) q[4];
rz(-0.036771688295259966) q[4];
ry(3.1411243799024944) q[5];
rz(2.5335673175903177) q[5];
ry(-0.052778493819819694) q[6];
rz(2.9572225609477623) q[6];
ry(0.0005086684448292189) q[7];
rz(2.704332557777069) q[7];
ry(3.006849918090255) q[8];
rz(3.1111002115474555) q[8];
ry(0.13009111403792417) q[9];
rz(0.1912482578721698) q[9];
ry(3.1412251917044545) q[10];
rz(2.5329618255565993) q[10];
ry(-0.004791670212506633) q[11];
rz(1.146036659344229) q[11];
ry(1.5024556479131386) q[12];
rz(0.42808466657545724) q[12];
ry(-1.3994293437811756) q[13];
rz(0.6036373919067346) q[13];
ry(-2.999185770098116) q[14];
rz(1.46379833556011) q[14];
ry(3.0801954955725934) q[15];
rz(2.4401830341769806) q[15];
ry(-0.010645621895784885) q[16];
rz(2.5315139372393425) q[16];
ry(3.134241891156058) q[17];
rz(1.7657707819548774) q[17];
ry(-0.9063951624010311) q[18];
rz(-1.9349682422786554) q[18];
ry(1.7422796575632473) q[19];
rz(2.7633190179438833) q[19];
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
ry(-0.007209886013181803) q[0];
rz(1.287991910479506) q[0];
ry(-0.01245269750855311) q[1];
rz(-0.8286404826950358) q[1];
ry(3.14143832234638) q[2];
rz(1.9589269408259777) q[2];
ry(-1.5585046352618213) q[3];
rz(2.5446605721015336) q[3];
ry(1.551246495933837) q[4];
rz(0.9475189113745596) q[4];
ry(3.070678637370036) q[5];
rz(-1.9006230778779003) q[5];
ry(-3.0837523868028724) q[6];
rz(-2.4311088001289756) q[6];
ry(3.0754964682559605) q[7];
rz(-0.37948015989729544) q[7];
ry(-1.5438507226865177) q[8];
rz(-0.647866348111696) q[8];
ry(1.5629970698075137) q[9];
rz(2.5846190479062168) q[9];
ry(1.5907522823743445) q[10];
rz(-0.6479457235107119) q[10];
ry(1.621701549720361) q[11];
rz(2.590441343912744) q[11];
ry(1.572629723886069) q[12];
rz(2.501022418880116) q[12];
ry(-1.560902530001859) q[13];
rz(-0.5737947342237719) q[13];
ry(1.5709476567865546) q[14];
rz(2.4315439757048156) q[14];
ry(-1.6023451655621281) q[15];
rz(2.5404851413306324) q[15];
ry(0.8955310215633336) q[16];
rz(-2.2624402708961737) q[16];
ry(-0.765760763159194) q[17];
rz(-2.286763602948346) q[17];
ry(-2.8066031082388174) q[18];
rz(-2.3173750287889354) q[18];
ry(-1.6318037434192973) q[19];
rz(-0.4264683689649448) q[19];