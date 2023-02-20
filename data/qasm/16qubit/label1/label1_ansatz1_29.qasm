OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-1.0302875028499683) q[0];
rz(-1.5585747419382416) q[0];
ry(-0.14892668568877984) q[1];
rz(-0.49605281325244094) q[1];
ry(-0.06452080860334154) q[2];
rz(-1.828919445647692) q[2];
ry(-2.3944373962006025) q[3];
rz(-2.0907442949769317) q[3];
ry(1.6687794090356376) q[4];
rz(0.014518923838361352) q[4];
ry(0.5100809350672484) q[5];
rz(1.0411935820506981) q[5];
ry(-0.13815290185027382) q[6];
rz(1.657406784251764) q[6];
ry(2.625608159048195) q[7];
rz(-3.070343390059814) q[7];
ry(-0.41196294827108204) q[8];
rz(2.6305183304295165) q[8];
ry(3.0381682380253996) q[9];
rz(1.9218841855742468) q[9];
ry(-2.428687209695026) q[10];
rz(2.919959645571428) q[10];
ry(-0.3312172610730668) q[11];
rz(-2.3450650659807617) q[11];
ry(-0.011829939160715952) q[12];
rz(-1.9968025905390911) q[12];
ry(-2.8726420266426853) q[13];
rz(1.712674302106184) q[13];
ry(-0.7718252823159677) q[14];
rz(0.5341093540859572) q[14];
ry(0.09382555071247543) q[15];
rz(2.525135636769839) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.8296325366705455) q[0];
rz(2.9172131055375172) q[0];
ry(1.281724394930178) q[1];
rz(-1.639130110528751) q[1];
ry(-0.8631816892913253) q[2];
rz(-2.1951100249068434) q[2];
ry(3.0078122289234277) q[3];
rz(1.0935682689303974) q[3];
ry(-0.005940288722930689) q[4];
rz(-2.985173897733301) q[4];
ry(3.1286614184149526) q[5];
rz(0.5259429049411607) q[5];
ry(2.7490773851210606) q[6];
rz(1.5812013634220514) q[6];
ry(-2.561219879963615) q[7];
rz(0.65683450831325) q[7];
ry(1.3213483173329807) q[8];
rz(-0.25859982790437647) q[8];
ry(2.9195318116499576) q[9];
rz(1.8394808755897374) q[9];
ry(2.7233119711748737) q[10];
rz(-0.5515264405726845) q[10];
ry(-0.3228539141829318) q[11];
rz(-2.6623726873155458) q[11];
ry(-3.006170303450385) q[12];
rz(1.6960896065544313) q[12];
ry(-1.857976314597276) q[13];
rz(-1.3593987299368935) q[13];
ry(0.8963950874171747) q[14];
rz(0.9671797354860292) q[14];
ry(0.31331887696662797) q[15];
rz(1.0859478154602389) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.5658297173712663) q[0];
rz(1.0289550159822705) q[0];
ry(-2.7479517721334066) q[1];
rz(0.2765810622041936) q[1];
ry(-0.023855553231601156) q[2];
rz(-1.0792830690226873) q[2];
ry(-2.997585200744602) q[3];
rz(-0.17855331587065135) q[3];
ry(-1.5026606725884222) q[4];
rz(1.5949942477179355) q[4];
ry(2.3713209708696175) q[5];
rz(-2.3320607427701123) q[5];
ry(3.029835096688509) q[6];
rz(-1.166562775935513) q[6];
ry(-0.7854710920858965) q[7];
rz(0.7593496646737559) q[7];
ry(-2.770212579525235) q[8];
rz(2.1356901878201624) q[8];
ry(3.010218895655488) q[9];
rz(-0.9197201277259266) q[9];
ry(2.651682996561584) q[10];
rz(-0.05699833237752959) q[10];
ry(-2.987949009925135) q[11];
rz(-1.9803649800545304) q[11];
ry(-1.4567603571021015) q[12];
rz(-0.3424637191062678) q[12];
ry(0.0844920959026082) q[13];
rz(0.8376370421308378) q[13];
ry(1.8555346692526316) q[14];
rz(0.756887618423817) q[14];
ry(1.8965925142678828) q[15];
rz(-0.8029408640407827) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.8734372901756497) q[0];
rz(-1.8240808514427331) q[0];
ry(-1.7916963131091432) q[1];
rz(-2.9186211327219107) q[1];
ry(2.825873634853018) q[2];
rz(-0.29663349427363084) q[2];
ry(-0.8053599714637247) q[3];
rz(1.3354553356247516) q[3];
ry(3.121209843470419) q[4];
rz(1.8984768528956355) q[4];
ry(0.17703616966020486) q[5];
rz(2.3247335086526526) q[5];
ry(-1.4247371538395461) q[6];
rz(-2.221304389938183) q[6];
ry(0.9774893554401665) q[7];
rz(3.086278092941733) q[7];
ry(2.279763702126196) q[8];
rz(-0.46638301745093885) q[8];
ry(0.6882535000691286) q[9];
rz(2.9241674082468236) q[9];
ry(-3.059733542185553) q[10];
rz(-2.7859382528360115) q[10];
ry(1.6253355921208825) q[11];
rz(-0.03545573084262315) q[11];
ry(-0.02378393682049218) q[12];
rz(-2.6667596562973794) q[12];
ry(0.12161436340251175) q[13];
rz(-0.11913112714646344) q[13];
ry(1.9573013062608373) q[14];
rz(-1.7822782146804905) q[14];
ry(-1.7511938903595494) q[15];
rz(-0.6655746383341318) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.44457879385188637) q[0];
rz(-1.881859678296782) q[0];
ry(2.683591865276815) q[1];
rz(-2.9645138204626345) q[1];
ry(-2.5851758084265977) q[2];
rz(2.8023149226514414) q[2];
ry(-2.8343599856077675) q[3];
rz(3.131380442473824) q[3];
ry(2.8464824349824243) q[4];
rz(-2.447826332398274) q[4];
ry(1.359036823068996) q[5];
rz(1.587206276372654) q[5];
ry(0.8997380047349806) q[6];
rz(-0.7575474896817714) q[6];
ry(-2.079879938861) q[7];
rz(2.4092159233493575) q[7];
ry(0.44296950130182916) q[8];
rz(-0.1375677877101736) q[8];
ry(2.7932470049530536) q[9];
rz(0.7197175326744852) q[9];
ry(-0.007808823252373859) q[10];
rz(1.9268506414791728) q[10];
ry(-0.40810177003559733) q[11];
rz(-3.080317377380471) q[11];
ry(2.5687446966211454) q[12];
rz(-2.720559604982958) q[12];
ry(0.40257843783110664) q[13];
rz(2.995119026380696) q[13];
ry(-2.3060039668787646) q[14];
rz(-2.218146197313901) q[14];
ry(2.297217811073237) q[15];
rz(-1.1300521144465108) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.391033518514478) q[0];
rz(0.2921728781946049) q[0];
ry(0.2810811130647952) q[1];
rz(0.6769392011063402) q[1];
ry(-2.7555600270642318) q[2];
rz(2.070512254949019) q[2];
ry(-2.9323850496835857) q[3];
rz(-0.5699160087310897) q[3];
ry(-3.0276014867024448) q[4];
rz(1.6048091011344998) q[4];
ry(-2.6947579835447275) q[5];
rz(0.5179264132786097) q[5];
ry(-1.4077478968347608) q[6];
rz(1.32864581994937) q[6];
ry(-2.6492110618149107) q[7];
rz(2.338950346297316) q[7];
ry(-0.4806410639143408) q[8];
rz(0.1408242396719678) q[8];
ry(-0.32087205113852146) q[9];
rz(-0.6542374098444378) q[9];
ry(3.1198513575076467) q[10];
rz(0.6558713792731584) q[10];
ry(1.4384215643186433) q[11];
rz(-2.2478939035092482) q[11];
ry(-3.0018462621193076) q[12];
rz(2.016785049319647) q[12];
ry(-1.164692544509954) q[13];
rz(-0.021089703920087466) q[13];
ry(0.2709470003535504) q[14];
rz(-2.547189774476858) q[14];
ry(-2.394020966047058) q[15];
rz(1.2619600102600839) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.8109515298814841) q[0];
rz(0.9339146373441585) q[0];
ry(0.44568183016660173) q[1];
rz(2.673983523261958) q[1];
ry(-2.381995860320911) q[2];
rz(2.170210225442541) q[2];
ry(-2.461890183752854) q[3];
rz(2.8133822881906516) q[3];
ry(-1.5504945229275648) q[4];
rz(-0.4201714582852407) q[4];
ry(-0.22585016787394518) q[5];
rz(-2.28511530819378) q[5];
ry(3.0611175171389045) q[6];
rz(0.03562727700573909) q[6];
ry(-2.510618151915257) q[7];
rz(-0.539069699024293) q[7];
ry(-2.7188423445571512) q[8];
rz(-2.3783527068321364) q[8];
ry(1.7758195246075401) q[9];
rz(1.3431837847070538) q[9];
ry(0.21226651943724129) q[10];
rz(-2.6684887897338365) q[10];
ry(-2.456040496588985) q[11];
rz(0.05930483198022874) q[11];
ry(1.6323339914602437) q[12];
rz(1.348266688482373) q[12];
ry(2.952904312744427) q[13];
rz(-1.0534193436321841) q[13];
ry(1.543407949117954) q[14];
rz(1.1036773003862181) q[14];
ry(2.294731836417576) q[15];
rz(2.9079302741866386) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.2142846456863587) q[0];
rz(0.761094601219198) q[0];
ry(-2.361650863116913) q[1];
rz(-2.108458980936825) q[1];
ry(-1.455232908008351) q[2];
rz(1.373914160079953) q[2];
ry(3.100289000177563) q[3];
rz(0.05910940448330759) q[3];
ry(-3.08984041082876) q[4];
rz(-3.092013543546095) q[4];
ry(3.09567469898085) q[5];
rz(1.858972488702653) q[5];
ry(0.6705759790863791) q[6];
rz(0.12739145615569114) q[6];
ry(2.1066353156171562) q[7];
rz(-2.390255570762206) q[7];
ry(-1.728656441619882) q[8];
rz(-1.6988642031892922) q[8];
ry(-3.0896174253576296) q[9];
rz(2.585763308597815) q[9];
ry(-0.5889005055981236) q[10];
rz(0.7259518795831136) q[10];
ry(0.02103459379852701) q[11];
rz(-3.0323051117207034) q[11];
ry(-0.031000544895312752) q[12];
rz(-0.21039933338383232) q[12];
ry(1.8558980258506894) q[13];
rz(-0.116691583434669) q[13];
ry(0.5499513691362353) q[14];
rz(2.2395625427031742) q[14];
ry(-0.5159269733895003) q[15];
rz(-2.1317777001161797) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.5353598524308554) q[0];
rz(-2.115672381776098) q[0];
ry(-0.058710032059349854) q[1];
rz(-1.0171086211149731) q[1];
ry(1.7458563988561813) q[2];
rz(0.27489785437114406) q[2];
ry(-1.0167932787570406) q[3];
rz(-1.977218515047687) q[3];
ry(1.2883102257121761) q[4];
rz(2.254283437796948) q[4];
ry(-3.105165139863303) q[5];
rz(-0.31097399304184487) q[5];
ry(1.7022609700692435) q[6];
rz(1.9574913237793217) q[6];
ry(0.37830831840122164) q[7];
rz(-3.048571334158765) q[7];
ry(2.7684431391222537) q[8];
rz(-1.5507538895450041) q[8];
ry(-3.1257133099378684) q[9];
rz(-1.540011591393287) q[9];
ry(-0.2609451902014397) q[10];
rz(-2.519093380876056) q[10];
ry(-2.2556954569139003) q[11];
rz(-1.1457475121640268) q[11];
ry(0.22876687258475759) q[12];
rz(-2.6895079321220763) q[12];
ry(-0.2416044178575101) q[13];
rz(-2.6182499918964854) q[13];
ry(0.4771973163827212) q[14];
rz(-1.0246125764746263) q[14];
ry(0.31168060035647027) q[15];
rz(0.55768113398847) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.9070573396799124) q[0];
rz(-1.9664661989401788) q[0];
ry(0.6610327171062808) q[1];
rz(-1.1223481476206854) q[1];
ry(0.2021338469605123) q[2];
rz(-0.41276499818667) q[2];
ry(3.1326300272002743) q[3];
rz(1.3671429133098374) q[3];
ry(-0.004301942858742969) q[4];
rz(2.9336833339616666) q[4];
ry(2.983925070620724) q[5];
rz(2.3590232582896418) q[5];
ry(-2.8125481960577288) q[6];
rz(-2.6058604840552237) q[6];
ry(0.927362774642476) q[7];
rz(-1.6624316672015176) q[7];
ry(1.4222807273523022) q[8];
rz(2.361497917981353) q[8];
ry(3.0442066711196136) q[9];
rz(-2.718286557733229) q[9];
ry(-1.3014944337626542) q[10];
rz(-0.6881789910255707) q[10];
ry(-0.05558196933239756) q[11];
rz(-3.0089733309548006) q[11];
ry(-3.136894503471603) q[12];
rz(-2.345733473767253) q[12];
ry(-2.07672166952677) q[13];
rz(-1.421678193759101) q[13];
ry(2.9731242926367187) q[14];
rz(-2.645517387801118) q[14];
ry(0.7328578789906959) q[15];
rz(-2.150116880891371) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.664069566519469) q[0];
rz(-2.8645162704221883) q[0];
ry(0.10648527231665028) q[1];
rz(2.008285136478768) q[1];
ry(-0.6738428840907976) q[2];
rz(-0.46151938013275545) q[2];
ry(2.373752863622226) q[3];
rz(0.17033579473794713) q[3];
ry(1.2652999250052586) q[4];
rz(-1.9301688386627387) q[4];
ry(3.128616054848602) q[5];
rz(-2.0783708847245013) q[5];
ry(1.7029341855168723) q[6];
rz(-2.482607232913089) q[6];
ry(0.35388416190205074) q[7];
rz(-1.1298922405052663) q[7];
ry(0.04986760893572843) q[8];
rz(3.065336704634626) q[8];
ry(3.007015059130214) q[9];
rz(2.492593392877865) q[9];
ry(-0.6399416776125717) q[10];
rz(-2.418119037643179) q[10];
ry(-0.5404756421784969) q[11];
rz(-1.0461285273025647) q[11];
ry(2.8290440771515346) q[12];
rz(2.936903817828305) q[12];
ry(0.6316671488275871) q[13];
rz(-0.7252352338867554) q[13];
ry(-2.716226122010526) q[14];
rz(-1.969511810667376) q[14];
ry(-1.5395662453814702) q[15];
rz(2.6201792072335506) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.279478523785718) q[0];
rz(0.953955314506515) q[0];
ry(-1.0750420076527378) q[1];
rz(2.3759003260644116) q[1];
ry(-2.141748903094322) q[2];
rz(-0.4150197928505224) q[2];
ry(-0.13373875794052323) q[3];
rz(2.3751548679471037) q[3];
ry(2.1700819927159163) q[4];
rz(1.4824740231638607) q[4];
ry(2.916058701572402) q[5];
rz(0.10627145466434568) q[5];
ry(0.350019376192634) q[6];
rz(0.07199950904198467) q[6];
ry(-1.0681992447712787) q[7];
rz(2.9359211003342796) q[7];
ry(-0.14140557003019438) q[8];
rz(-2.974362509723409) q[8];
ry(-3.0596047046449963) q[9];
rz(-2.5595289175362725) q[9];
ry(-0.43392637085670516) q[10];
rz(0.29218210615298795) q[10];
ry(0.0331489209287712) q[11];
rz(-2.94844983965177) q[11];
ry(-0.021789119784244804) q[12];
rz(1.4628391427790923) q[12];
ry(-1.1549459367304749) q[13];
rz(0.36449783150964254) q[13];
ry(-2.459459849732962) q[14];
rz(2.8404251919713293) q[14];
ry(-1.6252928483681268) q[15];
rz(2.8342910882420442) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.215044350569965) q[0];
rz(-1.383809872384358) q[0];
ry(-2.948270449000455) q[1];
rz(1.2811318329166168) q[1];
ry(-0.13756060841736328) q[2];
rz(2.64389726938432) q[2];
ry(2.8878969757727257) q[3];
rz(2.8338188357637004) q[3];
ry(-1.0414784886789656) q[4];
rz(-2.1486437759458683) q[4];
ry(-2.0533242927825306) q[5];
rz(1.0326076501507009) q[5];
ry(-1.2241220138831412) q[6];
rz(0.2104501522071152) q[6];
ry(-2.1943266701703497) q[7];
rz(1.7095815578462839) q[7];
ry(-3.1221697796788925) q[8];
rz(-0.10697709201562677) q[8];
ry(-1.1237870153923528) q[9];
rz(-1.1444742875496445) q[9];
ry(-1.7472024453485355) q[10];
rz(1.7578153936348444) q[10];
ry(3.0329696236950587) q[11];
rz(2.1431713607650913) q[11];
ry(2.8595421127727163) q[12];
rz(-0.6108904423146171) q[12];
ry(-0.2704912899397281) q[13];
rz(-1.6818120496542885) q[13];
ry(-1.8702298079234065) q[14];
rz(-2.5215627609281115) q[14];
ry(-0.2544539864725657) q[15];
rz(-1.5996567790725216) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.8608146396506908) q[0];
rz(0.5013958207134088) q[0];
ry(-2.862215656188778) q[1];
rz(-0.4555261933209307) q[1];
ry(2.5946951077335374) q[2];
rz(-0.9457709765140353) q[2];
ry(-0.03768421749716744) q[3];
rz(-2.1710661934793416) q[3];
ry(-0.0551532943691404) q[4];
rz(-2.2113087905565685) q[4];
ry(-0.002081146652355522) q[5];
rz(-1.1014024393545023) q[5];
ry(-3.131926914191746) q[6];
rz(-0.9935034692623589) q[6];
ry(2.074645133798748) q[7];
rz(1.4649719393188725) q[7];
ry(0.010298308351098238) q[8];
rz(1.4119436690614853) q[8];
ry(-0.009935396713425249) q[9];
rz(-2.082661860323885) q[9];
ry(-0.05145404831284283) q[10];
rz(0.08860589513086836) q[10];
ry(-3.109871681159734) q[11];
rz(1.1749764204686262) q[11];
ry(0.040523413260432406) q[12];
rz(3.0060895697034224) q[12];
ry(1.4482873321305165) q[13];
rz(2.7624857611471567) q[13];
ry(-1.005238476460202) q[14];
rz(2.458350005918727) q[14];
ry(0.07841267140502478) q[15];
rz(1.2045823452172124) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.8185608889639888) q[0];
rz(-1.113350681367904) q[0];
ry(-3.035199558397193) q[1];
rz(1.2805635011390288) q[1];
ry(3.0076955613460727) q[2];
rz(-2.9734656000595674) q[2];
ry(-2.843513636981375) q[3];
rz(1.493791343995974) q[3];
ry(0.1523032447692695) q[4];
rz(1.163838629686218) q[4];
ry(-1.9389862544562655) q[5];
rz(-2.175092631239038) q[5];
ry(2.3579994992544715) q[6];
rz(2.4566457080125446) q[6];
ry(2.7856147845662327) q[7];
rz(0.7493470170813659) q[7];
ry(-3.1063353792900736) q[8];
rz(2.227641508127975) q[8];
ry(-1.370057312919644) q[9];
rz(-0.3261874668835496) q[9];
ry(0.8769264105470473) q[10];
rz(2.743003444437793) q[10];
ry(0.6545978498859437) q[11];
rz(-2.2575959379799286) q[11];
ry(1.1405589374667935) q[12];
rz(3.0360307057054015) q[12];
ry(-1.2781111994742693) q[13];
rz(-0.6703364555950397) q[13];
ry(1.2964778146938851) q[14];
rz(-0.4625938606381247) q[14];
ry(1.969383468717966) q[15];
rz(2.108335640067912) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.9317328692401166) q[0];
rz(-0.6125410511537079) q[0];
ry(-0.3828300682862169) q[1];
rz(1.6361527267914306) q[1];
ry(-2.5671164980547765) q[2];
rz(-2.7831697086074314) q[2];
ry(0.05036367153003152) q[3];
rz(2.9302727784250475) q[3];
ry(-2.9050290221916453) q[4];
rz(0.21789392709445893) q[4];
ry(-0.006310948747048073) q[5];
rz(-0.1857818235756561) q[5];
ry(-2.188453608937744) q[6];
rz(2.1495766972461707) q[6];
ry(-1.62676953034905) q[7];
rz(-1.6280972646485528) q[7];
ry(-3.1342359067408636) q[8];
rz(0.17781357208468176) q[8];
ry(3.0926579462916104) q[9];
rz(2.6569573725297837) q[9];
ry(2.571596357434593) q[10];
rz(-2.850463791247183) q[10];
ry(3.04865638567675) q[11];
rz(-1.6848568542135973) q[11];
ry(2.8027643280723065) q[12];
rz(2.9252910439529214) q[12];
ry(-2.544694401258368) q[13];
rz(-1.6330475624886218) q[13];
ry(1.6034668696636454) q[14];
rz(-0.1419675899057283) q[14];
ry(-0.2404000624956626) q[15];
rz(-1.4830882856992453) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.8163937751303516) q[0];
rz(-0.7925963218047576) q[0];
ry(-1.7652035211283281) q[1];
rz(0.15136519793191994) q[1];
ry(0.21188261265218136) q[2];
rz(-0.07682611482026323) q[2];
ry(0.569010663245823) q[3];
rz(2.053008039095573) q[3];
ry(2.651200913451296) q[4];
rz(1.7600134794216027) q[4];
ry(-3.1229480057338916) q[5];
rz(-2.1029293654849908) q[5];
ry(2.7347946661450555) q[6];
rz(2.23199207690742) q[6];
ry(3.0070333271356846) q[7];
rz(-0.8592906864313239) q[7];
ry(-0.762242561811358) q[8];
rz(-2.9295777357101915) q[8];
ry(-0.30853425124429107) q[9];
rz(3.140424776656672) q[9];
ry(2.0505558266343664) q[10];
rz(-1.5644466518584526) q[10];
ry(-0.19589862825076665) q[11];
rz(2.261884616906299) q[11];
ry(-2.187508731110454) q[12];
rz(-0.28840609159544556) q[12];
ry(0.032725279402061026) q[13];
rz(2.092364034952071) q[13];
ry(1.4728022398840341) q[14];
rz(2.9255334670061437) q[14];
ry(-0.2172823239959616) q[15];
rz(-3.056691080316341) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.8019544245850634) q[0];
rz(2.6917432426015524) q[0];
ry(1.4086474807837934) q[1];
rz(-1.0536470972367535) q[1];
ry(-2.3680213838680015) q[2];
rz(-0.9289490815391224) q[2];
ry(-0.5399740233444897) q[3];
rz(-1.8732453532517481) q[3];
ry(2.204106893495357) q[4];
rz(1.67821686032035) q[4];
ry(0.3911426884758704) q[5];
rz(-0.5831983270901686) q[5];
ry(-1.6622110600240794) q[6];
rz(-0.7476769927066607) q[6];
ry(0.679438392582066) q[7];
rz(-2.0571033207493734) q[7];
ry(-2.075675859410184) q[8];
rz(0.40141489237586314) q[8];
ry(1.1268987194259856) q[9];
rz(-0.2221074356580072) q[9];
ry(-0.5452901503643295) q[10];
rz(-2.76561149237309) q[10];
ry(3.072019966679929) q[11];
rz(-1.9621441173534295) q[11];
ry(-0.4634080055722103) q[12];
rz(-2.442941761591335) q[12];
ry(3.0575956427449555) q[13];
rz(-1.2719862128234904) q[13];
ry(1.5903588667518442) q[14];
rz(-1.1811316736302677) q[14];
ry(3.0702544831833647) q[15];
rz(1.8924263975162408) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.5611773283028467) q[0];
rz(-2.3969192899075167) q[0];
ry(0.036956452162237516) q[1];
rz(-1.0225099246454719) q[1];
ry(-0.004745755214276709) q[2];
rz(0.9159598908720089) q[2];
ry(0.7870255699001653) q[3];
rz(-2.413198451363314) q[3];
ry(-0.046059316107828825) q[4];
rz(-1.1610920882601168) q[4];
ry(1.661196517729871) q[5];
rz(0.2141546021243699) q[5];
ry(1.6319908915292602) q[6];
rz(1.3299700799635417) q[6];
ry(1.3460916821278364) q[7];
rz(-1.7052847819525216) q[7];
ry(-0.1293783050353987) q[8];
rz(2.7395136291055855) q[8];
ry(2.561799464372476) q[9];
rz(-1.0994219272687964) q[9];
ry(0.1423097053710464) q[10];
rz(3.1269997726144663) q[10];
ry(-3.057305301970053) q[11];
rz(-0.7098694904833902) q[11];
ry(-0.9102919753372299) q[12];
rz(-1.2413467932508513) q[12];
ry(-1.62669238007844) q[13];
rz(-1.8788427592951653) q[13];
ry(-1.2959611422120447) q[14];
rz(-0.2186207313033527) q[14];
ry(2.065089098981482) q[15];
rz(-1.9874074898218477) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.2624254230484171) q[0];
rz(2.510591424831859) q[0];
ry(1.4921013222070911) q[1];
rz(-2.8509014440025027) q[1];
ry(-2.9435327887577305) q[2];
rz(-2.0163384784272216) q[2];
ry(-2.1319037800620113) q[3];
rz(-1.0685079051305184) q[3];
ry(2.332894334526829) q[4];
rz(0.0769161503207494) q[4];
ry(-0.1757725711448905) q[5];
rz(-1.5460081722121988) q[5];
ry(-3.140088280154096) q[6];
rz(-2.540999337087095) q[6];
ry(2.9060196102261457) q[7];
rz(2.518236164646961) q[7];
ry(0.08701610576446406) q[8];
rz(-2.954125373622233) q[8];
ry(-0.1137664528684152) q[9];
rz(1.0049631933296055) q[9];
ry(0.12053729973664849) q[10];
rz(0.18369488820352767) q[10];
ry(2.2642087402780158) q[11];
rz(1.3896966264611494) q[11];
ry(-0.35165673097147826) q[12];
rz(0.04777659789606492) q[12];
ry(2.482682398399457) q[13];
rz(2.984665018361239) q[13];
ry(-2.0955575273969265) q[14];
rz(2.5916256090786565) q[14];
ry(-0.078242891852236) q[15];
rz(-2.9705511088517595) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.5735670913248658) q[0];
rz(0.273075243651955) q[0];
ry(-1.766928580371264) q[1];
rz(-0.04628729318040659) q[1];
ry(-2.837719701548184) q[2];
rz(0.6211255382783297) q[2];
ry(2.7334600846606687) q[3];
rz(0.25152865712090455) q[3];
ry(-1.3820199761164682) q[4];
rz(-0.7126686279953446) q[4];
ry(1.2131678934217698) q[5];
rz(-2.193391843363193) q[5];
ry(-2.4880834882237943) q[6];
rz(-1.2115357852890574) q[6];
ry(-2.079227520777563) q[7];
rz(-1.9501736831346461) q[7];
ry(2.827659360614291) q[8];
rz(-2.0946640476984344) q[8];
ry(1.4817453446417632) q[9];
rz(3.0501862782921987) q[9];
ry(-2.955494089378572) q[10];
rz(1.1410949382931632) q[10];
ry(0.03875678501545554) q[11];
rz(-2.3785292390246315) q[11];
ry(-2.37606830637191) q[12];
rz(-3.0182956698170678) q[12];
ry(1.7548438683108025) q[13];
rz(0.6945932080911036) q[13];
ry(2.5696977437620254) q[14];
rz(2.418195867583321) q[14];
ry(-0.11401199012483049) q[15];
rz(2.269617887245496) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.6916010476816423) q[0];
rz(1.5772197774086285) q[0];
ry(1.2234495465344244) q[1];
rz(-0.0375445199671953) q[1];
ry(-3.0578378336608365) q[2];
rz(-2.3262672994279687) q[2];
ry(-2.768417146416978) q[3];
rz(-0.26804782663606724) q[3];
ry(2.6759228283323497) q[4];
rz(-0.757635350224918) q[4];
ry(3.0035565541896454) q[5];
rz(-1.2243876352506256) q[5];
ry(0.008698913601511647) q[6];
rz(0.17695140963642775) q[6];
ry(-2.864350715002184) q[7];
rz(-1.189143917408738) q[7];
ry(2.9934797318026205) q[8];
rz(-2.6817409267391614) q[8];
ry(1.0858071572992385) q[9];
rz(0.16254952561151725) q[9];
ry(0.002764288243225721) q[10];
rz(2.05888995700706) q[10];
ry(-0.02491260031678805) q[11];
rz(-2.1018970836859605) q[11];
ry(-1.859271517009927) q[12];
rz(1.6295519214380962) q[12];
ry(-1.231508498097913) q[13];
rz(0.7904927701766447) q[13];
ry(-1.0410406506089762) q[14];
rz(0.9277829533084407) q[14];
ry(1.3210019596890596) q[15];
rz(-2.1415695773548773) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.627844467921326) q[0];
rz(-0.31122696521675497) q[0];
ry(2.9661677892400897) q[1];
rz(-1.6048564191177181) q[1];
ry(-0.6864783221356809) q[2];
rz(1.4340198421374728) q[2];
ry(-1.6556323242093638) q[3];
rz(1.8436574136935306) q[3];
ry(-0.3098277068427864) q[4];
rz(0.08778828625947456) q[4];
ry(-2.8277676247258485) q[5];
rz(1.9932950338118784) q[5];
ry(0.13408151698797166) q[6];
rz(2.087224608721037) q[6];
ry(2.8796247054581805) q[7];
rz(-2.858688075025148) q[7];
ry(1.296293831879161) q[8];
rz(3.087296324845116) q[8];
ry(-2.400949607908441) q[9];
rz(-0.01009141125478) q[9];
ry(-1.7653231985216824) q[10];
rz(0.0366430366690552) q[10];
ry(0.3794821643628206) q[11];
rz(-3.0240017677678157) q[11];
ry(-0.030233031299815142) q[12];
rz(-1.3043278035807633) q[12];
ry(0.037938281291082454) q[13];
rz(-0.5392117472507872) q[13];
ry(2.0380592947628573) q[14];
rz(2.6768129477068396) q[14];
ry(0.7766070925625606) q[15];
rz(1.9429642797206887) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.3547200119526592) q[0];
rz(1.1208088835985852) q[0];
ry(2.4100921061103353) q[1];
rz(-1.328398320774265) q[1];
ry(1.7628798968245625) q[2];
rz(1.5067450512980305) q[2];
ry(0.3718581001099377) q[3];
rz(-2.3594084281505414) q[3];
ry(-0.48607044190998927) q[4];
rz(-2.2776915713851915) q[4];
ry(2.7509946556434297) q[5];
rz(-0.3438464391034031) q[5];
ry(0.013286649769206986) q[6];
rz(-1.7570024617716435) q[6];
ry(0.06095224818068168) q[7];
rz(-1.0645709441010738) q[7];
ry(0.1663249205015518) q[8];
rz(-0.004079849051336548) q[8];
ry(-0.02340245677692021) q[9];
rz(-1.4855766114433604) q[9];
ry(1.6144449929610598) q[10];
rz(1.3464888012535883) q[10];
ry(0.7892595060366562) q[11];
rz(2.76064319985329) q[11];
ry(-2.203983608755432) q[12];
rz(-0.016170503572222827) q[12];
ry(3.1125351414426605) q[13];
rz(2.4180808069646527) q[13];
ry(-2.973639394162836) q[14];
rz(2.8019527333871155) q[14];
ry(-0.7460660959245189) q[15];
rz(-1.5767469992772778) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.6112920144047378) q[0];
rz(1.8612245597145636) q[0];
ry(-0.05870104070647318) q[1];
rz(0.10603127580493688) q[1];
ry(1.746090432762197) q[2];
rz(0.05763685525266759) q[2];
ry(-0.03792548302927745) q[3];
rz(-0.9964579784200859) q[3];
ry(3.1182358716833405) q[4];
rz(-2.2600961640187136) q[4];
ry(2.5410685648636404) q[5];
rz(-0.24875454384941437) q[5];
ry(-0.07299461251206818) q[6];
rz(2.8837920013398475) q[6];
ry(2.9345993242779103) q[7];
rz(0.8165371342660954) q[7];
ry(-1.2283060814452178) q[8];
rz(-1.9025662805335195) q[8];
ry(-1.7176007358042353) q[9];
rz(-0.7839605534308233) q[9];
ry(1.9114290093897952) q[10];
rz(-1.3726910265788672) q[10];
ry(-1.4881451157357244) q[11];
rz(-2.1005159744430815) q[11];
ry(2.696132227339109) q[12];
rz(2.433919097992866) q[12];
ry(-0.17231471151244193) q[13];
rz(-0.17861184455401247) q[13];
ry(2.1196233194135656) q[14];
rz(2.0729590604141572) q[14];
ry(1.1419661181041754) q[15];
rz(2.016668149585537) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.4604677980524101) q[0];
rz(-2.6898568945264496) q[0];
ry(-0.01406561729931255) q[1];
rz(-0.0744545727093842) q[1];
ry(1.7930043665344968) q[2];
rz(2.738737956050547) q[2];
ry(1.7244875289610144) q[3];
rz(0.3678371118919526) q[3];
ry(2.6130798057776463) q[4];
rz(3.1049769820996147) q[4];
ry(-0.22175325478653907) q[5];
rz(2.870314311042859) q[5];
ry(3.1301290389687995) q[6];
rz(0.5196076506933363) q[6];
ry(0.2588334568225735) q[7];
rz(-0.4219113729346037) q[7];
ry(-0.09758589435241216) q[8];
rz(1.786036191330675) q[8];
ry(-3.121301441460708) q[9];
rz(-1.9594553212107937) q[9];
ry(0.08626424987503079) q[10];
rz(1.3747751153313341) q[10];
ry(-3.110109253152541) q[11];
rz(-2.3012937732729952) q[11];
ry(-0.03217594193685488) q[12];
rz(-2.9035475830842277) q[12];
ry(0.848188933532519) q[13];
rz(1.0675687477320703) q[13];
ry(2.0604839520797675) q[14];
rz(-2.220903247543532) q[14];
ry(-1.4277910667437625) q[15];
rz(1.4118469967000724) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.177718431637797) q[0];
rz(0.45758466598545) q[0];
ry(-0.5845361693962999) q[1];
rz(-3.054219985579977) q[1];
ry(1.5026021065223096) q[2];
rz(2.052826180197128) q[2];
ry(-0.44121946788719196) q[3];
rz(0.1199234632103332) q[3];
ry(-2.3140802151529085) q[4];
rz(3.0183142358689232) q[4];
ry(-2.8574567793310095) q[5];
rz(-2.1251629125347615) q[5];
ry(-2.9332384305474344) q[6];
rz(0.9939704266978153) q[6];
ry(-1.5974546314377687) q[7];
rz(3.0616125626740107) q[7];
ry(0.5068381898334618) q[8];
rz(-0.8678581148764496) q[8];
ry(-1.6543334985198115) q[9];
rz(-1.235026244413005) q[9];
ry(-1.9464974197847822) q[10];
rz(1.8090495929813546) q[10];
ry(-0.8279358738497838) q[11];
rz(0.2227829389387007) q[11];
ry(0.3460605503728367) q[12];
rz(1.8886446932002796) q[12];
ry(-2.7995714834459724) q[13];
rz(2.7458788830910437) q[13];
ry(1.7903598126604887) q[14];
rz(2.3866944218528827) q[14];
ry(0.880916472265507) q[15];
rz(2.458807859681067) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.7951751186806737) q[0];
rz(0.09155672265395098) q[0];
ry(-1.6187981001124296) q[1];
rz(-2.0866896016159506) q[1];
ry(1.5987018923163216) q[2];
rz(0.011371273631016088) q[2];
ry(2.8975404168880763) q[3];
rz(-1.1882449933738348) q[3];
ry(0.1317115533676168) q[4];
rz(-2.7656041885222065) q[4];
ry(-0.7746376590349594) q[5];
rz(-3.0267406706585622) q[5];
ry(0.012511933851366584) q[6];
rz(-0.18632491694218523) q[6];
ry(2.663586724763415) q[7];
rz(0.7260289012207178) q[7];
ry(0.01246398179974939) q[8];
rz(1.5723752471678305) q[8];
ry(-0.016624427614234882) q[9];
rz(-1.8665436015480106) q[9];
ry(0.16145647788536915) q[10];
rz(1.552254101559682) q[10];
ry(-0.5717371232737705) q[11];
rz(-2.8195538287749233) q[11];
ry(3.043744402464889) q[12];
rz(1.791119380115715) q[12];
ry(0.6640692752572086) q[13];
rz(2.7274458972033697) q[13];
ry(0.4925826006778414) q[14];
rz(-0.044340536835125725) q[14];
ry(0.9870866137200976) q[15];
rz(1.4115720468606048) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.752398950333946) q[0];
rz(-2.738068595838851) q[0];
ry(0.02159808648849193) q[1];
rz(-1.76392960997355) q[1];
ry(-1.5590730311055818) q[2];
rz(1.8741123477960444) q[2];
ry(1.6144208704896355) q[3];
rz(2.449158871440942) q[3];
ry(3.087166899631289) q[4];
rz(2.22344708056764) q[4];
ry(-2.120812507919719) q[5];
rz(-3.0750800440803183) q[5];
ry(0.016423369429530835) q[6];
rz(1.7804073098436772) q[6];
ry(1.9698671906717513) q[7];
rz(-1.3076153199890008) q[7];
ry(2.312601094432108) q[8];
rz(-0.5924693711372306) q[8];
ry(1.468688426332146) q[9];
rz(1.9824612183713157) q[9];
ry(1.5316017149882477) q[10];
rz(-0.4272291736450872) q[10];
ry(0.750112618415964) q[11];
rz(-1.7787327852531325) q[11];
ry(-2.2747817423407444) q[12];
rz(0.007407967582675035) q[12];
ry(-0.03208770865463012) q[13];
rz(0.9649686209158237) q[13];
ry(2.5295727198722107) q[14];
rz(2.486659941741353) q[14];
ry(-1.8550101063906599) q[15];
rz(0.0008289032958884514) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.5597518339552696) q[0];
rz(-2.244066425118363) q[0];
ry(-2.6177923450066456) q[1];
rz(3.0703579360669147) q[1];
ry(-2.867032888508962) q[2];
rz(-3.0541505057792997) q[2];
ry(-3.0966336312860547) q[3];
rz(-2.966230558344429) q[3];
ry(0.04502852442832772) q[4];
rz(0.6312597473234032) q[4];
ry(2.447241028323254) q[5];
rz(1.995305495698776) q[5];
ry(-3.0695768861437434) q[6];
rz(1.5598396513759694) q[6];
ry(3.070101412899098) q[7];
rz(1.4993643644387251) q[7];
ry(3.046758192152948) q[8];
rz(1.376143080552139) q[8];
ry(-2.2796194251776507) q[9];
rz(-0.03563105106181652) q[9];
ry(0.038195009432961946) q[10];
rz(-0.8995582606201716) q[10];
ry(-0.18937118329809888) q[11];
rz(1.6308699727260672) q[11];
ry(-1.5609979428805696) q[12];
rz(-1.5396791781046146) q[12];
ry(-0.022489421776084963) q[13];
rz(3.03661089057396) q[13];
ry(-2.72239543561201) q[14];
rz(-0.6638466130672834) q[14];
ry(-2.076471606490874) q[15];
rz(2.897665171657753) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.8426551421181503) q[0];
rz(1.2479548430146117) q[0];
ry(0.6572835419582992) q[1];
rz(1.127007226655685) q[1];
ry(2.8724205386485258) q[2];
rz(-0.10497210301070825) q[2];
ry(-1.7195780314601299) q[3];
rz(1.4043926234228232) q[3];
ry(2.7283435247223227) q[4];
rz(-1.5590055227112707) q[4];
ry(0.5998855231404713) q[5];
rz(-0.3183629845342883) q[5];
ry(1.4010462681237221) q[6];
rz(-3.0892343567696927) q[6];
ry(-0.7445397386351342) q[7];
rz(2.412575249900578) q[7];
ry(3.1312310834892068) q[8];
rz(-1.7394444229636026) q[8];
ry(-0.11609668550689169) q[9];
rz(-3.1107212978632273) q[9];
ry(3.1353330985858086) q[10];
rz(0.40580513289639514) q[10];
ry(0.45775177776695486) q[11];
rz(1.5216189527379471) q[11];
ry(1.5285083102263093) q[12];
rz(0.15642737436290663) q[12];
ry(-1.563785444564835) q[13];
rz(-1.5736815620793159) q[13];
ry(-1.6064935197258698) q[14];
rz(0.05243349849886303) q[14];
ry(3.0240704920553734) q[15];
rz(-2.667047287599983) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.2261050831517384) q[0];
rz(-2.5170909723683246) q[0];
ry(2.9886939181476313) q[1];
rz(2.680364810237548) q[1];
ry(6.729715555164672e-05) q[2];
rz(-0.634087220062935) q[2];
ry(3.0549974349197955) q[3];
rz(-1.2525190701529259) q[3];
ry(-0.07252188986629164) q[4];
rz(2.5708052808229374) q[4];
ry(3.100536737352142) q[5];
rz(3.1090216417753003) q[5];
ry(3.135624993184862) q[6];
rz(2.6760510616519584) q[6];
ry(-3.097241748100865) q[7];
rz(1.2880201979834078) q[7];
ry(-0.15471787345966262) q[8];
rz(-0.03676727639616363) q[8];
ry(0.8543847563649258) q[9];
rz(-2.21421477663705) q[9];
ry(-0.040843247086443095) q[10];
rz(-3.0017730813363546) q[10];
ry(-3.1334043070834245) q[11];
rz(-3.129616953158334) q[11];
ry(-3.1402085991251365) q[12];
rz(-1.5224601138164369) q[12];
ry(3.016650254603546) q[13];
rz(-3.139653232081177) q[13];
ry(-1.5684178195236704) q[14];
rz(1.5715666764253697) q[14];
ry(0.1925087676710413) q[15];
rz(-2.9153825028803597) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.13871253346575194) q[0];
rz(0.46519463705575337) q[0];
ry(-1.587209269894321) q[1];
rz(2.8548376450233564) q[1];
ry(1.836438220818235) q[2];
rz(-0.6549108275600029) q[2];
ry(-1.1451716461857793) q[3];
rz(0.2741137367422298) q[3];
ry(-1.417635972399042) q[4];
rz(-0.540952345627086) q[4];
ry(0.6757203969708468) q[5];
rz(2.589200761696259) q[5];
ry(2.8171093364620026) q[6];
rz(-2.189589600945877) q[6];
ry(1.988343834516936) q[7];
rz(2.3280635134239516) q[7];
ry(-0.9969368083459462) q[8];
rz(2.439333359289561) q[8];
ry(0.0011552763785243059) q[9];
rz(2.2036217539620475) q[9];
ry(-3.1408448628265884) q[10];
rz(-1.2456865777072599) q[10];
ry(-1.601477841943803) q[11];
rz(-1.1942849339180543) q[11];
ry(0.008593184403867365) q[12];
rz(1.1682803294457873) q[12];
ry(-1.5698786594651044) q[13];
rz(0.16234997976419407) q[13];
ry(1.5709074252542043) q[14];
rz(-3.055188308312359) q[14];
ry(-0.0024833073812912635) q[15];
rz(-0.2406248485438326) q[15];