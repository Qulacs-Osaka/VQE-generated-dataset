OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(0.017358111395185116) q[0];
rz(-2.2599048820381924) q[0];
ry(0.016984169181403056) q[1];
rz(-1.0457924725032877) q[1];
ry(1.5229256022667625) q[2];
rz(3.1032619843942792) q[2];
ry(-1.5669982276429455) q[3];
rz(-2.350436851611973) q[3];
ry(-2.7938861838272544e-05) q[4];
rz(-2.6299726203644624) q[4];
ry(-3.1415805128009002) q[5];
rz(2.9906588840006862) q[5];
ry(-3.140612957626418) q[6];
rz(-1.9836722469845594) q[6];
ry(3.094954150560976) q[7];
rz(-3.1166281583150344) q[7];
ry(-1.546294710215661) q[8];
rz(3.138720225934207) q[8];
ry(-1.5680687709640546) q[9];
rz(3.1412618656369298) q[9];
ry(3.141391029966009) q[10];
rz(0.574452054388395) q[10];
ry(-0.00032537999544195916) q[11];
rz(-1.1670981744533793) q[11];
ry(-3.141288954727871) q[12];
rz(0.2980741775783704) q[12];
ry(-3.1338164722686916) q[13];
rz(0.0273531113910126) q[13];
ry(-1.5707419718639832) q[14];
rz(2.719789236309868) q[14];
ry(1.5707584855633254) q[15];
rz(2.9833688777629273) q[15];
ry(3.1409614581143623) q[16];
rz(-0.7557016686059732) q[16];
ry(0.014008412796170242) q[17];
rz(1.6050835047663483) q[17];
ry(-3.1394765344137947) q[18];
rz(0.6067458594521489) q[18];
ry(0.00021473562587140776) q[19];
rz(2.646670258579353) q[19];
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
ry(3.1377180818247496) q[0];
rz(-2.0449010997276504) q[0];
ry(-2.4242093653035024) q[1];
rz(-0.06050739408074347) q[1];
ry(1.5724037946357798) q[2];
rz(-0.7948074048072201) q[2];
ry(-3.1298022200473814) q[3];
rz(0.7183955953410169) q[3];
ry(-3.0943307468054493) q[4];
rz(-0.4388845438744083) q[4];
ry(3.1414430626898002) q[5];
rz(-1.1582574448439966) q[5];
ry(-1.5830298183750706) q[6];
rz(-1.276147663695304) q[6];
ry(-0.6974873778002557) q[7];
rz(-0.006664471309594688) q[7];
ry(0.11450554594994546) q[8];
rz(1.8028850255759217) q[8];
ry(1.4624332954469828) q[9];
rz(1.0301117265761532) q[9];
ry(2.7606288832599635) q[10];
rz(-2.2245143593646457) q[10];
ry(-1.9968992215656183) q[11];
rz(1.9081495474704784) q[11];
ry(-1.5695922003502858) q[12];
rz(-1.5247963249786318) q[12];
ry(1.7270110784927102) q[13];
rz(2.057430757215638) q[13];
ry(0.3630594432619851) q[14];
rz(-2.7571542823829245) q[14];
ry(1.9443430293598898) q[15];
rz(0.023815098564891106) q[15];
ry(1.5689574963644546) q[16];
rz(-2.48397648929761) q[16];
ry(-1.5747960119335902) q[17];
rz(1.5335121088008725) q[17];
ry(0.005943699714447526) q[18];
rz(-1.1585492941581395) q[18];
ry(-0.00033581379048966756) q[19];
rz(-0.48859068439176306) q[19];
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
ry(1.5677648964426596) q[0];
rz(3.0981858373152993) q[0];
ry(2.3314001128024304) q[1];
rz(-2.005625545431777) q[1];
ry(3.1363979836079694) q[2];
rz(-1.1562490537629517) q[2];
ry(0.004408380083577901) q[3];
rz(0.2645751684825868) q[3];
ry(0.001735173516074028) q[4];
rz(1.2826798164413655) q[4];
ry(3.0776173160826716) q[5];
rz(0.8339521013239735) q[5];
ry(0.008560643683241055) q[6];
rz(-1.5043713408637716) q[6];
ry(-1.5717060092426396) q[7];
rz(-1.6171874814260103) q[7];
ry(3.1400551589258874) q[8];
rz(0.9504455798211726) q[8];
ry(0.0018052701319446708) q[9];
rz(1.238325785845578) q[9];
ry(-0.0006588730565528778) q[10];
rz(-0.1133912572566896) q[10];
ry(-0.011148973924998806) q[11];
rz(2.2618460812991072) q[11];
ry(3.1399829356114184) q[12];
rz(0.040972337037096906) q[12];
ry(0.010266763888625796) q[13];
rz(2.824354806865507) q[13];
ry(-1.3900064719464809) q[14];
rz(1.4143385498277308) q[14];
ry(1.8116624659782403) q[15];
rz(-1.7255324382630977) q[15];
ry(2.185644412505122) q[16];
rz(-3.0519698805105087) q[16];
ry(-0.9459621660331765) q[17];
rz(-2.228455164102602) q[17];
ry(1.5714746104804656) q[18];
rz(-3.0125162383184354) q[18];
ry(1.569139426297968) q[19];
rz(1.4981502704959309) q[19];
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
ry(1.583730242693215) q[0];
rz(2.8123683127783403) q[0];
ry(-0.03817824246202268) q[1];
rz(-0.8321156697647247) q[1];
ry(2.998088780169105) q[2];
rz(-1.0284605733958394) q[2];
ry(-1.7040797898276336) q[3];
rz(-1.6021247111340688) q[3];
ry(-3.13251942594895) q[4];
rz(0.9533620736717326) q[4];
ry(-1.5793686593584653) q[5];
rz(-1.8100853417855687) q[5];
ry(0.13523731976990444) q[6];
rz(1.8648817744316872) q[6];
ry(1.5267805624932853) q[7];
rz(1.0215277954063682) q[7];
ry(-1.0463051999352029) q[8];
rz(-0.5933473275558305) q[8];
ry(1.3832370198758899) q[9];
rz(2.389838869735553) q[9];
ry(1.2984017288197807) q[10];
rz(2.8772354771787367) q[10];
ry(2.398230752904816) q[11];
rz(2.9450710586595186) q[11];
ry(-0.7278965039806751) q[12];
rz(1.5661347420592688) q[12];
ry(0.7403003090409784) q[13];
rz(-1.8027389603521504) q[13];
ry(1.5455842734933032) q[14];
rz(3.073716574292998) q[14];
ry(1.5633444515399262) q[15];
rz(-0.022370034801762403) q[15];
ry(1.5687374111630672) q[16];
rz(0.22027423573821636) q[16];
ry(-1.5749029814550284) q[17];
rz(-0.1412102020430872) q[17];
ry(-0.051057965760120005) q[18];
rz(0.25463845949380204) q[18];
ry(1.5264801687253167) q[19];
rz(-1.520342315521586) q[19];
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
ry(1.5155613252859537) q[0];
rz(2.6225325869599447) q[0];
ry(1.6308165530080228) q[1];
rz(2.5271804130928945) q[1];
ry(0.05107390307504467) q[2];
rz(1.7457531034821088) q[2];
ry(2.969790430634533) q[3];
rz(2.2728189839590653) q[3];
ry(-3.1382632858885073) q[4];
rz(0.8849241667897798) q[4];
ry(-3.1414113969130106) q[5];
rz(3.128140537766008) q[5];
ry(-0.00014233407781816257) q[6];
rz(-3.0074309287603893) q[6];
ry(0.0034063585994354995) q[7];
rz(-1.9206728453569568) q[7];
ry(-2.9860143298096116) q[8];
rz(1.3026977362745198) q[8];
ry(0.10404589927646933) q[9];
rz(2.9413040581721743) q[9];
ry(-3.0399334372328757) q[10];
rz(0.005662985421998279) q[10];
ry(-2.913102394902427) q[11];
rz(-1.120316189705009) q[11];
ry(1.578351819661239) q[12];
rz(0.3769680684268444) q[12];
ry(-1.5639979641322639) q[13];
rz(2.0816016689667975) q[13];
ry(1.5795064431106944) q[14];
rz(-1.3690476034549441) q[14];
ry(-1.5574432794391067) q[15];
rz(1.7588016226098746) q[15];
ry(-3.1267012933483183) q[16];
rz(-1.388155453985193) q[16];
ry(-3.1159470106594003) q[17];
rz(-1.7059905743587898) q[17];
ry(0.000981782573964196) q[18];
rz(-1.8013923942441838) q[18];
ry(-1.57291601326976) q[19];
rz(1.5196158823923387) q[19];
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
ry(0.11472958955831736) q[0];
rz(-2.153375704549827) q[0];
ry(-0.1154871730020903) q[1];
rz(2.4104763980273543) q[1];
ry(0.03842798186443819) q[2];
rz(2.0942001030405013) q[2];
ry(-2.824934409634711) q[3];
rz(2.28789273492627) q[3];
ry(1.5619795919172752) q[4];
rz(-1.9722262622976174) q[4];
ry(3.1352712409568078) q[5];
rz(2.527947250227319) q[5];
ry(-1.6141044889431946) q[6];
rz(-0.11417238278778559) q[6];
ry(0.7902219708236702) q[7];
rz(1.9717945223990085) q[7];
ry(-2.7900111075110163) q[8];
rz(-0.8521827769584671) q[8];
ry(0.36654741071993513) q[9];
rz(-1.8616688645834323) q[9];
ry(1.57041240869375) q[10];
rz(1.5912319051579455) q[10];
ry(-1.583250182599235) q[11];
rz(-0.18947801278220577) q[11];
ry(3.0904513833873795) q[12];
rz(1.8902430690879353) q[12];
ry(3.1405832492979457) q[13];
rz(-2.4199511239590286) q[13];
ry(0.37803925207037903) q[14];
rz(-1.770873791421991) q[14];
ry(0.36791563326337334) q[15];
rz(0.5815913424247569) q[15];
ry(-1.5923275491914008) q[16];
rz(-2.197350891948348) q[16];
ry(1.6026861978555897) q[17];
rz(2.4730339268923194) q[17];
ry(0.005486498558682138) q[18];
rz(-0.7824304233151523) q[18];
ry(1.5730946442179805) q[19];
rz(1.7596947006643218) q[19];
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
ry(1.0081299746513857) q[0];
rz(-0.49889869591198555) q[0];
ry(-1.0922905767599893) q[1];
rz(0.9839481840521014) q[1];
ry(0.00043055426727586934) q[2];
rz(-0.23061391543713206) q[2];
ry(1.5698846547002925) q[3];
rz(-1.8969376527716766) q[3];
ry(-3.1282414257359927) q[4];
rz(1.3260662538605903) q[4];
ry(0.004110962353155223) q[5];
rz(-2.2075341930455) q[5];
ry(3.1104136207102377) q[6];
rz(2.9518239301749807) q[6];
ry(0.016787551774307907) q[7];
rz(-0.827364711494254) q[7];
ry(-1.5815034603321032) q[8];
rz(1.6169580647670585) q[8];
ry(1.5287076455602753) q[9];
rz(1.4941563441328178) q[9];
ry(0.3374519809407896) q[10];
rz(2.876697720682555) q[10];
ry(0.000694022570853825) q[11];
rz(1.0237641374091042) q[11];
ry(-0.1531761816873578) q[12];
rz(-1.666221441611046) q[12];
ry(-1.3467046977958814) q[13];
rz(1.1818011628389569) q[13];
ry(0.7375937952649014) q[14];
rz(2.8898772627005513) q[14];
ry(-0.0416598411369673) q[15];
rz(-0.7921725434374421) q[15];
ry(-1.5350459668572443) q[16];
rz(-3.0612150811304275) q[16];
ry(-1.6091241221845536) q[17];
rz(-0.4372937941257688) q[17];
ry(1.421757154496973) q[18];
rz(1.4017064074994778) q[18];
ry(2.509816120345527) q[19];
rz(-2.7928177594804082) q[19];
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
ry(0.0009802477542628241) q[0];
rz(1.5320258383143628) q[0];
ry(3.1415049272575017) q[1];
rz(-1.1446922653276213) q[1];
ry(0.00036760422471954023) q[2];
rz(1.3723927859703449) q[2];
ry(0.002094065373833054) q[3];
rz(-1.5862735218562478) q[3];
ry(3.061379308265453) q[4];
rz(1.2758279413915705) q[4];
ry(-1.5579996714342101) q[5];
rz(-2.4909118531902426) q[5];
ry(-0.018718422382190347) q[6];
rz(0.02315778575345505) q[6];
ry(-0.020269402116131374) q[7];
rz(0.542255484524027) q[7];
ry(2.6294000685123287) q[8];
rz(2.542275298866916) q[8];
ry(2.7321768248523695) q[9];
rz(2.803825175500176) q[9];
ry(0.001312837184771709) q[10];
rz(0.23092683581297296) q[10];
ry(0.049954719912990164) q[11];
rz(3.135963625273294) q[11];
ry(0.007237529363408768) q[12];
rz(-1.4147342665175882) q[12];
ry(-0.013664715750016931) q[13];
rz(-1.9667449221522055) q[13];
ry(0.0006100048951273461) q[14];
rz(-2.8770769588727076) q[14];
ry(-3.139198642270692) q[15];
rz(2.1361668367789934) q[15];
ry(-3.137940305884082) q[16];
rz(0.07504640003357614) q[16];
ry(3.1363778687092503) q[17];
rz(-0.019630794962738563) q[17];
ry(3.1415508659641787) q[18];
rz(-2.114736675720762) q[18];
ry(1.499601216957661) q[19];
rz(0.8595512836808901) q[19];
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
ry(-1.6399280906098284) q[0];
rz(-1.274217996011819) q[0];
ry(0.011175973999727695) q[1];
rz(1.9139735058322325) q[1];
ry(1.5850979081409762) q[2];
rz(0.4464393273598901) q[2];
ry(-3.1413781128394116) q[3];
rz(-1.9084084188590325) q[3];
ry(3.140724418068239) q[4];
rz(-0.4561863079078345) q[4];
ry(-3.1386083937796725) q[5];
rz(-2.494034964609405) q[5];
ry(2.355102393383967) q[6];
rz(1.5634936026757948) q[6];
ry(0.7764485616757915) q[7];
rz(1.5645574347270097) q[7];
ry(-3.1231804104463463) q[8];
rz(2.542452263628847) q[8];
ry(0.007743218895854831) q[9];
rz(-2.832241879180071) q[9];
ry(1.7203518123939856) q[10];
rz(0.0013682283757789904) q[10];
ry(1.5626920919020597) q[11];
rz(-0.7941607078588455) q[11];
ry(-1.4294928993634084) q[12];
rz(1.365764032482427) q[12];
ry(0.3145963801097196) q[13];
rz(0.12179821028510851) q[13];
ry(-2.4125016679006746) q[14];
rz(-0.2755747871098366) q[14];
ry(-0.030206528224886273) q[15];
rz(2.689686948228866) q[15];
ry(-0.0410577474550399) q[16];
rz(-3.06813829720151) q[16];
ry(0.004166544760308475) q[17];
rz(2.552546962403214) q[17];
ry(-0.016340480408962854) q[18];
rz(-1.3358717414632073) q[18];
ry(3.1181881182607096) q[19];
rz(0.8629768449283342) q[19];
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
ry(0.005726889282756101) q[0];
rz(1.6054865107182827) q[0];
ry(-0.004154109079316202) q[1];
rz(-1.629003442082885) q[1];
ry(3.1370891482331817) q[2];
rz(-1.534296387251958) q[2];
ry(-1.571855505304463) q[3];
rz(-0.4140391972736843) q[3];
ry(-1.5736758098956674) q[4];
rz(-2.004223020596486) q[4];
ry(-1.5737175746971734) q[5];
rz(-1.9877696554799027) q[5];
ry(-2.3695857491085857) q[6];
rz(-2.0683128454914983) q[6];
ry(2.360865608541173) q[7];
rz(1.0735928151856795) q[7];
ry(-1.5629184065588932) q[8];
rz(2.6296223207969818) q[8];
ry(1.563514790671797) q[9];
rz(-0.5123751722976104) q[9];
ry(-1.5813855344640997) q[10];
rz(-0.5103312267416449) q[10];
ry(-3.122830058942453) q[11];
rz(1.8375562912216246) q[11];
ry(-0.029193435880515928) q[12];
rz(-1.8991796739111655) q[12];
ry(0.031179358031167975) q[13];
rz(1.686900149864141) q[13];
ry(-1.548220934100925) q[14];
rz(1.0422784516747035) q[14];
ry(-1.5479610660432437) q[15];
rz(1.0411244500849257) q[15];
ry(-1.5920763338058253) q[16];
rz(-2.1004657912490785) q[16];
ry(-1.5496800908087371) q[17];
rz(1.0383864636815705) q[17];
ry(0.1946581028430554) q[18];
rz(-0.41440948966493435) q[18];
ry(-1.6899004116416751) q[19];
rz(2.7268776890369213) q[19];