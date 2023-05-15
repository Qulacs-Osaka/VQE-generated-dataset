OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-1.2813347065854925) q[0];
rz(-2.7282908570152675) q[0];
ry(-0.6996910408090988) q[1];
rz(-0.15986554272736458) q[1];
ry(-0.002830709275760235) q[2];
rz(0.7377002966428479) q[2];
ry(-3.137049978890794) q[3];
rz(0.16168276349993074) q[3];
ry(-1.5785017282095977) q[4];
rz(0.9980064927874291) q[4];
ry(1.577747358685938) q[5];
rz(1.9971792591941462) q[5];
ry(-3.140363706460567) q[6];
rz(-2.4995315066216617) q[6];
ry(-0.0042407389489225586) q[7];
rz(-2.8822498424515994) q[7];
ry(0.003979808753307168) q[8];
rz(-1.0381019598161147) q[8];
ry(-0.00567182432892821) q[9];
rz(-1.2458402984681518) q[9];
ry(-1.5390332852074422) q[10];
rz(2.552928502054253) q[10];
ry(-1.7222231054726298) q[11];
rz(-2.7196805147885015) q[11];
ry(-3.1412342503673845) q[12];
rz(-2.7970767695349608) q[12];
ry(0.0002390843286540252) q[13];
rz(-1.3302835000383895) q[13];
ry(-1.5262426216480183) q[14];
rz(3.0961227686851154) q[14];
ry(-1.5655745466929574) q[15];
rz(-1.5046063031805073) q[15];
ry(-0.0015800782070609645) q[16];
rz(-1.3885206316605103) q[16];
ry(0.0012266776678080404) q[17];
rz(3.1305316595189487) q[17];
ry(-1.5488195022539997) q[18];
rz(-1.8860251574498097) q[18];
ry(1.5802436544162488) q[19];
rz(-0.5491307585609083) q[19];
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
ry(-0.3589527914979467) q[0];
rz(-0.07186464655689508) q[0];
ry(2.913266499845032) q[1];
rz(1.905921401472352) q[1];
ry(0.027105567894063043) q[2];
rz(-1.3237998759006127) q[2];
ry(3.139033813181242) q[3];
rz(2.3372463367356313) q[3];
ry(1.9338038235207433) q[4];
rz(1.024024665581286) q[4];
ry(2.1045492589333223) q[5];
rz(2.883317277575774) q[5];
ry(-2.3166496341774874) q[6];
rz(-1.3262822729738257) q[6];
ry(0.37618974049781845) q[7];
rz(-1.7477657324433489) q[7];
ry(1.5423491921893777) q[8];
rz(1.2259607472874914) q[8];
ry(-1.53891380920519) q[9];
rz(3.015314578846962) q[9];
ry(-0.5838654908951177) q[10];
rz(1.7180945431858916) q[10];
ry(1.99724748489301) q[11];
rz(-1.2876111828641763) q[11];
ry(-0.0016609363062987147) q[12];
rz(1.6340367220514453) q[12];
ry(3.140007844423663) q[13];
rz(0.2011435008382216) q[13];
ry(1.2153447530010606) q[14];
rz(2.59360106680045) q[14];
ry(2.565723646230832) q[15];
rz(1.3386970299958678) q[15];
ry(-2.844476912074676) q[16];
rz(1.004503687511514) q[16];
ry(-2.2743755536478245) q[17];
rz(1.6127775564890445) q[17];
ry(-2.5479955198639566) q[18];
rz(2.7859872534163497) q[18];
ry(1.3938657073105505) q[19];
rz(-0.6875454658408406) q[19];
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
ry(0.9054871669213416) q[0];
rz(-2.498372283577297) q[0];
ry(-1.623691283449637) q[1];
rz(0.705932387882325) q[1];
ry(1.4994391455995704) q[2];
rz(1.8104486341623973) q[2];
ry(1.6302408587403883) q[3];
rz(1.362152573352927) q[3];
ry(-0.07623882744618626) q[4];
rz(2.8305972696770154) q[4];
ry(-0.07602549514411905) q[5];
rz(-0.1734743886604977) q[5];
ry(0.0033938300293403947) q[6];
rz(-3.1185348320377315) q[6];
ry(0.024069082055876656) q[7];
rz(-0.08071669132954273) q[7];
ry(0.001163842579828869) q[8];
rz(2.5022457462954852) q[8];
ry(-0.012730091320424929) q[9];
rz(1.478996346764899) q[9];
ry(-1.3769533725681482) q[10];
rz(-1.5537015638808613) q[10];
ry(0.23499352824708747) q[11];
rz(1.1243813999497112) q[11];
ry(3.141376252790889) q[12];
rz(-3.1380871695911994) q[12];
ry(3.1414177398502408) q[13];
rz(1.674294039865708) q[13];
ry(3.097350399213157) q[14];
rz(-2.8128171199853638) q[14];
ry(-3.1373853576062802) q[15];
rz(-2.6930693408119906) q[15];
ry(-3.094199981915407) q[16];
rz(1.790129759926109) q[16];
ry(-3.0214979727664533) q[17];
rz(-0.1506445236440053) q[17];
ry(0.03811942173088719) q[18];
rz(-2.0606918225654764) q[18];
ry(-2.8809550932249426) q[19];
rz(2.2568342079749337) q[19];
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
ry(-2.939005535632527) q[0];
rz(1.3732432315704974) q[0];
ry(0.30241530160805186) q[1];
rz(0.6356229391242235) q[1];
ry(0.06557629886617658) q[2];
rz(-0.330424351289917) q[2];
ry(0.01553374570881605) q[3];
rz(-2.979474199865008) q[3];
ry(-0.004025728954605271) q[4];
rz(2.2843375396670824) q[4];
ry(0.004610467940482366) q[5];
rz(-1.0636092038417084) q[5];
ry(1.7934273983644227) q[6];
rz(0.30961809999277184) q[6];
ry(1.6380648733585152) q[7];
rz(-1.1673645493809852) q[7];
ry(-0.008253165552285502) q[8];
rz(0.9777249092830945) q[8];
ry(-3.117004549661629) q[9];
rz(2.928788682715494) q[9];
ry(1.5390915586469331) q[10];
rz(-3.1053651826236957) q[10];
ry(-3.129085155204467) q[11];
rz(-0.1704909930151717) q[11];
ry(0.0009332618982602624) q[12];
rz(-1.6402122377766224) q[12];
ry(-0.0032630826717161554) q[13];
rz(1.7184208992172032) q[13];
ry(1.9183662776677095) q[14];
rz(-2.9033382199850193) q[14];
ry(2.47921688653465) q[15];
rz(-3.001040149801825) q[15];
ry(-0.06014753990836809) q[16];
rz(-1.913805630791227) q[16];
ry(0.16191882468787622) q[17];
rz(0.6853191923887278) q[17];
ry(0.10168254919569401) q[18];
rz(0.5005941784880719) q[18];
ry(-0.07760965019015091) q[19];
rz(-0.054536418527927934) q[19];
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
ry(-2.5457535141402605) q[0];
rz(-0.13727868211777317) q[0];
ry(2.7997206831861967) q[1];
rz(-2.422742865229734) q[1];
ry(-3.0615290097998265) q[2];
rz(-1.017880582036982) q[2];
ry(-2.202565858634769) q[3];
rz(1.090300909998961) q[3];
ry(2.482619869703421) q[4];
rz(-1.4655142016145624) q[4];
ry(-0.628378570300506) q[5];
rz(1.4526214060835834) q[5];
ry(-3.1365792380983093) q[6];
rz(1.6305788063143467) q[6];
ry(0.013798248344629007) q[7];
rz(-0.6837694862593611) q[7];
ry(1.5740396313735685) q[8];
rz(-1.489882905403718) q[8];
ry(-1.5665625383099195) q[9];
rz(1.647286739672779) q[9];
ry(-2.1133554926304488) q[10];
rz(-1.033343829753056) q[10];
ry(-1.4848765816367193) q[11];
rz(-2.897292245653191) q[11];
ry(3.1341533279229816) q[12];
rz(1.2955142284688095) q[12];
ry(-3.0941986723581905) q[13];
rz(-0.39119398908324204) q[13];
ry(-1.5448207543177104) q[14];
rz(-2.8908812138256192) q[14];
ry(-1.5534963282802652) q[15];
rz(-3.0031280167972114) q[15];
ry(-0.1270138888664557) q[16];
rz(-2.8122700617136505) q[16];
ry(-2.8170865394197673) q[17];
rz(-1.645955652053348) q[17];
ry(0.9187809451106485) q[18];
rz(1.5635568180821584) q[18];
ry(-0.7659189438596199) q[19];
rz(-1.3271696884842314) q[19];
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
ry(-0.6083757911920333) q[0];
rz(1.9770622500387578) q[0];
ry(1.809798665260531) q[1];
rz(1.695330442137779) q[1];
ry(-3.139847258927271) q[2];
rz(-2.6098534360253005) q[2];
ry(0.06974090882102413) q[3];
rz(0.6151131998257627) q[3];
ry(0.12104674819045524) q[4];
rz(0.35119092493780485) q[4];
ry(3.021151625093347) q[5];
rz(0.34386266928272224) q[5];
ry(-3.1016121281925138) q[6];
rz(-2.139292885964566) q[6];
ry(0.03775831009159702) q[7];
rz(-3.0068461019158805) q[7];
ry(-1.5942674105242565) q[8];
rz(1.3111814548469551) q[8];
ry(-1.6027550544738425) q[9];
rz(3.03472095174904) q[9];
ry(2.4266157490342044) q[10];
rz(-1.396563627830382) q[10];
ry(-2.391376402046452) q[11];
rz(-1.6656550232651781) q[11];
ry(-0.058964260978973115) q[12];
rz(-1.974069095075249) q[12];
ry(-2.96951234065277) q[13];
rz(-0.8967273723353006) q[13];
ry(2.4410784790195987) q[14];
rz(-0.07979184988453092) q[14];
ry(2.4566763259021873) q[15];
rz(2.9629835019700246) q[15];
ry(1.9292608347027027) q[16];
rz(-0.5065719331266684) q[16];
ry(0.6568518535073775) q[17];
rz(2.4175173037620286) q[17];
ry(-2.265228584232437) q[18];
rz(-1.9377461918851715) q[18];
ry(0.895519372383835) q[19];
rz(-0.9524687329045927) q[19];
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
ry(-1.5314317613558037) q[0];
rz(1.2653838732845615) q[0];
ry(2.9913479425208327) q[1];
rz(-1.4776685465544181) q[1];
ry(-1.6194310429957257) q[2];
rz(-2.9839542917282253) q[2];
ry(-1.5339548691491696) q[3];
rz(0.1811350921403738) q[3];
ry(-1.4252180144665827) q[4];
rz(0.5523930849052237) q[4];
ry(1.460495328652029) q[5];
rz(-1.1137321985520752) q[5];
ry(2.772366337458472) q[6];
rz(1.4935195125292733) q[6];
ry(-0.47614479522258507) q[7];
rz(-1.5377928083226287) q[7];
ry(1.0620660046993717) q[8];
rz(-3.070993347246664) q[8];
ry(-2.353732478928025) q[9];
rz(1.5444023425376452) q[9];
ry(-1.6954966353495857) q[10];
rz(3.0299926849076972) q[10];
ry(-1.4119542394166358) q[11];
rz(0.1639373115148285) q[11];
ry(0.48684887169360186) q[12];
rz(1.1355180821561413) q[12];
ry(-0.6409077068770765) q[13];
rz(2.2907846355164825) q[13];
ry(-0.6727997846490734) q[14];
rz(0.1893849252473103) q[14];
ry(0.6927629685150869) q[15];
rz(1.8649957864556725) q[15];
ry(0.736489633369116) q[16];
rz(-0.7427440767288279) q[16];
ry(-0.76210970705442) q[17];
rz(2.251489027992405) q[17];
ry(-2.9786515593436453) q[18];
rz(-1.8025953409411888) q[18];
ry(-2.978574053399269) q[19];
rz(0.28670018518197155) q[19];
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
ry(-0.18861417364443422) q[0];
rz(2.261317249130834) q[0];
ry(1.685849562099712) q[1];
rz(-2.1728171644044503) q[1];
ry(-1.5383113158566957) q[2];
rz(-1.4049974360709556) q[2];
ry(1.579105776090401) q[3];
rz(0.09724868841186404) q[3];
ry(0.003340803355338004) q[4];
rz(1.3473100687119974) q[4];
ry(-0.021284897830048617) q[5];
rz(3.0348336495703414) q[5];
ry(-3.103440378947589) q[6];
rz(-2.9116224477710575) q[6];
ry(0.044641481998363375) q[7];
rz(-1.9018426801492583) q[7];
ry(0.008444190824578259) q[8];
rz(-0.18585222985845373) q[8];
ry(-0.00047317237753663477) q[9];
rz(-1.9388860776772816) q[9];
ry(-3.1010352754626993) q[10];
rz(1.5194266412239052) q[10];
ry(3.100801067496789) q[11];
rz(-2.7038852388276142) q[11];
ry(-0.32112522017932793) q[12];
rz(-0.9257430569511261) q[12];
ry(-0.6068217276835145) q[13];
rz(-3.0126459162800736) q[13];
ry(-0.0057765478127628275) q[14];
rz(0.4054337829672539) q[14];
ry(-0.03828278215086112) q[15];
rz(-2.5947996529132795) q[15];
ry(0.4181120104234741) q[16];
rz(1.2488095410979594) q[16];
ry(-0.2567150163611181) q[17];
rz(-2.1678603514828785) q[17];
ry(0.1786053413945723) q[18];
rz(2.352780338046535) q[18];
ry(1.4585488438255456) q[19];
rz(-0.15289322802783428) q[19];
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
ry(-2.153484317467533) q[0];
rz(-3.132182537576928) q[0];
ry(-1.480179863651113) q[1];
rz(-0.9371111415970893) q[1];
ry(1.7771910028872968) q[2];
rz(0.9569450066169657) q[2];
ry(-2.581923412974625) q[3];
rz(2.20597975695883) q[3];
ry(1.6229439365822875) q[4];
rz(1.2680885839425613) q[4];
ry(1.5173034036659208) q[5];
rz(-0.2102329918717043) q[5];
ry(2.2477095238314124) q[6];
rz(0.6676931718534105) q[6];
ry(-2.0248276287656117) q[7];
rz(0.18712564007719096) q[7];
ry(-0.8994254387855598) q[8];
rz(-0.8225128916673023) q[8];
ry(-0.8749325265723273) q[9];
rz(-2.224439378054781) q[9];
ry(0.08521757379389783) q[10];
rz(2.0710596153458294) q[10];
ry(-3.0113928575229716) q[11];
rz(0.7848730342755229) q[11];
ry(1.2440081241029537) q[12];
rz(2.4607193481208207) q[12];
ry(1.7083686928540482) q[13];
rz(2.193924879236957) q[13];
ry(-3.1406837392635505) q[14];
rz(-1.1489820313691608) q[14];
ry(0.0031478844857790603) q[15];
rz(-0.6603947601748752) q[15];
ry(0.010783718310514506) q[16];
rz(-0.12642216618570104) q[16];
ry(0.04020637717138621) q[17];
rz(-2.7111213673807697) q[17];
ry(2.6955894947005987) q[18];
rz(2.9443658495977827) q[18];
ry(-0.09641984785900881) q[19];
rz(1.8203564314411862) q[19];
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
ry(-3.1023365308799877) q[0];
rz(-1.7764382880528977) q[0];
ry(-1.2993484455622388) q[1];
rz(-0.24452941074784285) q[1];
ry(-0.040788678740693037) q[2];
rz(-0.24731606489914673) q[2];
ry(-0.016032499254881927) q[3];
rz(0.4226046734519513) q[3];
ry(0.010596066796005216) q[4];
rz(-2.887190531442602) q[4];
ry(3.0640313130750054) q[5];
rz(-1.069013312397388) q[5];
ry(-0.8970484337515865) q[6];
rz(-1.8785048301855416) q[6];
ry(1.8150172251987429) q[7];
rz(1.1714527435462925) q[7];
ry(3.1184128776365734) q[8];
rz(-0.020236696472780463) q[8];
ry(-3.117322068169263) q[9];
rz(2.5713019486097224) q[9];
ry(-2.792397186508112) q[10];
rz(0.581609991986466) q[10];
ry(0.3318460715245992) q[11];
rz(0.29895489093528216) q[11];
ry(-2.7940028920617976) q[12];
rz(-2.6530568046066203) q[12];
ry(-2.9068053772587) q[13];
rz(1.493919422238128) q[13];
ry(1.4910014112335332) q[14];
rz(2.8770380276747956) q[14];
ry(-1.4823010903674847) q[15];
rz(0.2152784059852056) q[15];
ry(-0.036854399277241434) q[16];
rz(-2.2383675879926654) q[16];
ry(0.025729328975651278) q[17];
rz(0.22089632350130992) q[17];
ry(1.3871072315505142) q[18];
rz(-0.40059786066460307) q[18];
ry(0.8185414179393256) q[19];
rz(-1.5500214419642768) q[19];
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
ry(1.3945713412101006) q[0];
rz(2.0792664771456586) q[0];
ry(0.7932191953615089) q[1];
rz(-0.2470683409454608) q[1];
ry(1.7750571275200666) q[2];
rz(-1.0774213317476633) q[2];
ry(-0.723133037712473) q[3];
rz(2.2225686440166497) q[3];
ry(-0.0018174608865087901) q[4];
rz(2.161900725493465) q[4];
ry(-0.0037106428752302634) q[5];
rz(2.4731514252846654) q[5];
ry(-0.17287737516963322) q[6];
rz(2.447840038051765) q[6];
ry(2.849364599550887) q[7];
rz(2.2231573958744475) q[7];
ry(3.1214730312249652) q[8];
rz(1.8449697200897663) q[8];
ry(0.012690292388955804) q[9];
rz(0.0816533477006214) q[9];
ry(0.08259874105236421) q[10];
rz(-0.5298311431636911) q[10];
ry(-3.0833104227549133) q[11];
rz(0.2760655797907414) q[11];
ry(-2.1453310942705923) q[12];
rz(2.8743265523087764) q[12];
ry(2.2037412071863867) q[13];
rz(0.42669190917027666) q[13];
ry(1.4070435803806935) q[14];
rz(-3.018270105366845) q[14];
ry(-1.4079959913384372) q[15];
rz(-0.0322812760258273) q[15];
ry(0.7746262007869776) q[16];
rz(2.661940001043381) q[16];
ry(1.3843065127797867) q[17];
rz(0.30198766673744704) q[17];
ry(-0.1873429244881226) q[18];
rz(-0.2279355059265961) q[18];
ry(-0.19110444223362677) q[19];
rz(-1.3414580199520092) q[19];
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
ry(-2.695511100823916) q[0];
rz(2.035695627597819) q[0];
ry(2.7187492342414834) q[1];
rz(1.002336967206511) q[1];
ry(1.748202611565481) q[2];
rz(-1.7273649736972052) q[2];
ry(-1.820886621051609) q[3];
rz(1.6688083525194397) q[3];
ry(-3.0922437257326187) q[4];
rz(-2.73063236991819) q[4];
ry(-0.0003676201248508093) q[5];
rz(-2.8868426613736102) q[5];
ry(-1.8259347444350704) q[6];
rz(1.7188817574490203) q[6];
ry(0.5791961952219921) q[7];
rz(-2.6092342549658993) q[7];
ry(-3.084286534515532) q[8];
rz(2.972523450737749) q[8];
ry(-3.1401897037936104) q[9];
rz(1.348964600284642) q[9];
ry(-0.2797857537248535) q[10];
rz(-1.2701947295257394) q[10];
ry(-2.8893326268798636) q[11];
rz(0.6869660031909155) q[11];
ry(1.396185160003717) q[12];
rz(3.0926563280262496) q[12];
ry(-2.1247852231434816) q[13];
rz(1.4364910038316845) q[13];
ry(0.2701914463233175) q[14];
rz(1.0576346664598597) q[14];
ry(2.774614125831289) q[15];
rz(1.1506822670114607) q[15];
ry(-1.0814072739633465) q[16];
rz(0.644111701292112) q[16];
ry(0.5428785414354683) q[17];
rz(3.11310038037092) q[17];
ry(2.4270699123570165) q[18];
rz(1.8604237088495683) q[18];
ry(2.549350321989429) q[19];
rz(-2.0468162298217245) q[19];
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
ry(-3.0505823931397016) q[0];
rz(0.20217299349083628) q[0];
ry(3.08163939503974) q[1];
rz(-0.33850836350611685) q[1];
ry(2.53609569819749) q[2];
rz(-1.719464675194481) q[2];
ry(-1.0391616485541064) q[3];
rz(-0.0234272913430269) q[3];
ry(3.137812725010745) q[4];
rz(2.750369283681528) q[4];
ry(3.092878350771497) q[5];
rz(2.521933203835496) q[5];
ry(0.11425142591193183) q[6];
rz(-1.9953178659332398) q[6];
ry(-3.027255480106544) q[7];
rz(-1.1337981404609971) q[7];
ry(-0.03188302714338054) q[8];
rz(1.9115793049095624) q[8];
ry(-3.0986108620816113) q[9];
rz(-1.1901764280015836) q[9];
ry(-3.1112836794767476) q[10];
rz(-2.4695880035832216) q[10];
ry(0.0247555122287082) q[11];
rz(-2.7119899566615073) q[11];
ry(-0.03719686819713974) q[12];
rz(0.03879970899002781) q[12];
ry(-3.069616719221444) q[13];
rz(-2.502405070531673) q[13];
ry(1.260673724222697) q[14];
rz(-1.839133393814663) q[14];
ry(1.8971430415877109) q[15];
rz(-1.1987999966579252) q[15];
ry(0.016598608657417326) q[16];
rz(-2.216012978312855) q[16];
ry(0.003984615874933702) q[17];
rz(2.541289337078384) q[17];
ry(0.4226820123812267) q[18];
rz(-2.283018010227167) q[18];
ry(-2.7108050529461827) q[19];
rz(0.8754601781310387) q[19];
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
ry(-0.40054543177814694) q[0];
rz(-0.25548640399721284) q[0];
ry(-0.3724837253077884) q[1];
rz(-1.3402441259726405) q[1];
ry(0.582206142193499) q[2];
rz(1.9081342044954857) q[2];
ry(2.509443298306698) q[3];
rz(-0.9693502155360706) q[3];
ry(-2.967508670664284) q[4];
rz(-0.47071692920511676) q[4];
ry(-0.10076829515994223) q[5];
rz(2.2754073735564666) q[5];
ry(2.4415557990587584) q[6];
rz(1.193445975929369) q[6];
ry(1.098847214971598) q[7];
rz(0.3046848997615834) q[7];
ry(1.5856316608882757) q[8];
rz(-3.023124475452792) q[8];
ry(1.5415754258637095) q[9];
rz(1.0090170470623203) q[9];
ry(0.026261146949373426) q[10];
rz(-0.9108991483187268) q[10];
ry(0.0006247712771782217) q[11];
rz(-1.3005603755333768) q[11];
ry(-0.0769630076381569) q[12];
rz(2.8380833418527005) q[12];
ry(-0.18999972197085402) q[13];
rz(-2.061085999676434) q[13];
ry(-2.8634117814011666) q[14];
rz(1.6526700126885425) q[14];
ry(2.9375852279816934) q[15];
rz(-1.0691049723311092) q[15];
ry(2.3375293890428135) q[16];
rz(1.0833411081098534) q[16];
ry(0.808625270379137) q[17];
rz(-1.0860420766205734) q[17];
ry(2.4423199821541055) q[18];
rz(1.0586179505057058) q[18];
ry(-0.7968900068708554) q[19];
rz(-0.9987590917653023) q[19];
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
ry(0.3340864832572968) q[0];
rz(-2.520492824330688) q[0];
ry(-1.4439047858167662) q[1];
rz(1.2431773226160499) q[1];
ry(0.25013471677929267) q[2];
rz(-2.834584205174729) q[2];
ry(-2.9042668182035265) q[3];
rz(0.11891114956750039) q[3];
ry(2.3493284582188103) q[4];
rz(-1.8324311023443602) q[4];
ry(-0.7922445435138145) q[5];
rz(-2.20173041545514) q[5];
ry(-2.8591133421392114) q[6];
rz(0.944631375005384) q[6];
ry(0.265193951578075) q[7];
rz(-2.49281495006489) q[7];
ry(-3.1229060295990987) q[8];
rz(1.4587851323785195) q[8];
ry(-3.132422389895197) q[9];
rz(2.9580353168366513) q[9];
ry(-0.0028941815587000264) q[10];
rz(-1.8829234054081034) q[10];
ry(-0.044264505929440444) q[11];
rz(1.7196848975475016) q[11];
ry(2.767313052421441) q[12];
rz(2.966353940397655) q[12];
ry(2.4542586511786975) q[13];
rz(3.0288287096055653) q[13];
ry(-3.017013881087082) q[14];
rz(-1.3103039985454235) q[14];
ry(0.12186487537846119) q[15];
rz(-1.8042595387476348) q[15];
ry(1.6015756981208231) q[16];
rz(2.5990112689585967) q[16];
ry(-1.8029030538912156) q[17];
rz(-1.8825134577423348) q[17];
ry(-1.1744976594504903) q[18];
rz(1.5447211000181893) q[18];
ry(2.027176209723653) q[19];
rz(-1.5274398039131816) q[19];
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
ry(0.14548472458461226) q[0];
rz(0.5267633909283269) q[0];
ry(0.42849686081918037) q[1];
rz(0.5571820710587988) q[1];
ry(3.1207679529622427) q[2];
rz(2.729838309966224) q[2];
ry(3.1305692778829806) q[3];
rz(1.3744909763463746) q[3];
ry(-2.8160232917774555) q[4];
rz(-0.10723905433489912) q[4];
ry(3.0940617151051857) q[5];
rz(2.1015362970618288) q[5];
ry(-2.2003844973677307) q[6];
rz(-1.7130764534908174) q[6];
ry(2.454270813872266) q[7];
rz(2.574881641015659) q[7];
ry(2.630005727931649) q[8];
rz(-0.6469770315564363) q[8];
ry(-0.2932984562140263) q[9];
rz(-0.18455492543298746) q[9];
ry(-0.0008828030652514585) q[10];
rz(2.774335875445448) q[10];
ry(0.006404288434090875) q[11];
rz(1.1908142435057198) q[11];
ry(0.1394708836819812) q[12];
rz(2.006901315507361) q[12];
ry(3.102649270572276) q[13];
rz(1.673623731049247) q[13];
ry(2.4621825784490765) q[14];
rz(-2.7696043524164224) q[14];
ry(2.466000939280741) q[15];
rz(-2.147109350145835) q[15];
ry(-0.22896123658918643) q[16];
rz(3.1136416902687984) q[16];
ry(0.47820761140054086) q[17];
rz(-0.23399003855107559) q[17];
ry(-0.5574322181863808) q[18];
rz(0.5878041487710602) q[18];
ry(0.5595815870089762) q[19];
rz(-0.3555042097472025) q[19];
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
ry(2.0698798918455803) q[0];
rz(0.45589009523368246) q[0];
ry(1.0829938479253203) q[1];
rz(1.52553194563014) q[1];
ry(0.004638350722855782) q[2];
rz(-0.7035090245891453) q[2];
ry(-3.0493946163043266) q[3];
rz(-2.5104863514368425) q[3];
ry(0.7950846494500687) q[4];
rz(-2.948577161185516) q[4];
ry(0.9703667834674068) q[5];
rz(-1.2458020469112017) q[5];
ry(-3.141522374803599) q[6];
rz(2.0245651150125337) q[6];
ry(0.0003638086630211035) q[7];
rz(-1.345621438191697) q[7];
ry(-0.16642007120697055) q[8];
rz(2.0078602656495503) q[8];
ry(2.851838282948877) q[9];
rz(-1.2588450144534162) q[9];
ry(-0.009449865667797042) q[10];
rz(0.8887513829042794) q[10];
ry(3.07077691994709) q[11];
rz(-2.527615607505958) q[11];
ry(-1.7176179723801954) q[12];
rz(2.9073721758022737) q[12];
ry(1.801965182965148) q[13];
rz(2.3123234636686654) q[13];
ry(-3.0990135741752987) q[14];
rz(-2.9616590288300153) q[14];
ry(-3.0874370392970216) q[15];
rz(-2.268595622404774) q[15];
ry(1.4071760842137344) q[16];
rz(-1.482395392455131) q[16];
ry(-0.9285485104434044) q[17];
rz(-1.1444161729883395) q[17];
ry(-0.2589441872326485) q[18];
rz(2.3784977527256976) q[18];
ry(-0.2451169401219429) q[19];
rz(-0.8378722243195478) q[19];
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
ry(-0.8621093871024447) q[0];
rz(2.329941434082387) q[0];
ry(-2.998032854126303) q[1];
rz(2.445121001174064) q[1];
ry(-0.012317033213450998) q[2];
rz(-1.641050316318181) q[2];
ry(-3.129845239266521) q[3];
rz(-0.7878341446956888) q[3];
ry(-3.0135930343525352) q[4];
rz(-2.8348807997656107) q[4];
ry(-0.11722851739784695) q[5];
rz(1.5299710786496297) q[5];
ry(-3.1092245062163766) q[6];
rz(2.3976465015322685) q[6];
ry(0.02689671894912138) q[7];
rz(-2.981550565706276) q[7];
ry(-1.5854006690307625) q[8];
rz(2.718950588443276) q[8];
ry(-1.577779668541475) q[9];
rz(1.2806622938627286) q[9];
ry(1.211469814893857) q[10];
rz(-0.02024658902732135) q[10];
ry(-0.037730631733651876) q[11];
rz(2.475581277509996) q[11];
ry(3.1269122912351413) q[12];
rz(-3.0008867131100354) q[12];
ry(0.004535567802103714) q[13];
rz(-2.0631466181352627) q[13];
ry(0.128284345134408) q[14];
rz(2.1621955624468887) q[14];
ry(0.11275781642317195) q[15];
rz(2.1200052463728896) q[15];
ry(1.3393621643317353) q[16];
rz(0.17479367293871653) q[16];
ry(-1.4453676077441884) q[17];
rz(-0.11398494995569158) q[17];
ry(0.0032650825453259813) q[18];
rz(0.8783341336429825) q[18];
ry(0.0029725807459932696) q[19];
rz(0.4444445897213821) q[19];
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
ry(1.235428779367961) q[0];
rz(-2.312882658867134) q[0];
ry(0.853584506803497) q[1];
rz(2.152276856567708) q[1];
ry(0.016198660494903917) q[2];
rz(-1.9480054614499842) q[2];
ry(0.11707809928781246) q[3];
rz(-1.4784862753845254) q[3];
ry(-1.472184329451383) q[4];
rz(-1.4175339155132711) q[4];
ry(-1.5109178049629008) q[5];
rz(-1.9993969289943652) q[5];
ry(2.9596326286237895) q[6];
rz(-3.063803143007521) q[6];
ry(3.097445186061679) q[7];
rz(-1.3775531303885833) q[7];
ry(0.017026453106473313) q[8];
rz(1.062632742540897) q[8];
ry(3.0737501817326587) q[9];
rz(-0.4279669223261706) q[9];
ry(2.820670614574891) q[10];
rz(-1.5705310952315763) q[10];
ry(-0.08165868833198342) q[11];
rz(1.5093689544114466) q[11];
ry(3.0982995954546535) q[12];
rz(2.613208108870272) q[12];
ry(0.002306779556596972) q[13];
rz(-3.043355613866268) q[13];
ry(0.9022255510107753) q[14];
rz(-0.17706546672555867) q[14];
ry(0.8820772591788728) q[15];
rz(2.7378918755287707) q[15];
ry(2.0339827286044048) q[16];
rz(0.2212328526508081) q[16];
ry(-1.1120697752473854) q[17];
rz(0.5600098418743261) q[17];
ry(1.1113494917423) q[18];
rz(-2.304621058647555) q[18];
ry(-1.088538955738932) q[19];
rz(2.0445147450717496) q[19];
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
ry(-0.003072017207255584) q[0];
rz(-1.4681367036321928) q[0];
ry(-3.1390848432909753) q[1];
rz(-0.8031411859228639) q[1];
ry(0.3325067453788201) q[2];
rz(-1.7921256808123733) q[2];
ry(-0.33694400266147045) q[3];
rz(-2.652724111513953) q[3];
ry(-1.5275540778390557) q[4];
rz(0.9599483352315223) q[4];
ry(-1.6896575162151353) q[5];
rz(0.6244575836795674) q[5];
ry(0.015425422594663256) q[6];
rz(3.0056319542512506) q[6];
ry(0.0017934257312465763) q[7];
rz(1.2104373054916981) q[7];
ry(-0.007911444224819085) q[8];
rz(-2.5321744108627793) q[8];
ry(3.139494712018502) q[9];
rz(-0.5518643981356914) q[9];
ry(-1.5856941491719052) q[10];
rz(1.9083626564933118) q[10];
ry(-1.5576268991708915) q[11];
rz(3.1237996134835218) q[11];
ry(0.006308051225582271) q[12];
rz(0.29752566241968814) q[12];
ry(3.135845729265447) q[13];
rz(1.9542271966801377) q[13];
ry(-3.1289303900111554) q[14];
rz(-2.263394757462465) q[14];
ry(3.122796320111283) q[15];
rz(0.7246565328716237) q[15];
ry(-2.313100678323244) q[16];
rz(-2.954051108295004) q[16];
ry(0.9067973084853295) q[17];
rz(2.754359922950445) q[17];
ry(3.1202353254932427) q[18];
rz(-0.27442019539010243) q[18];
ry(-0.021555211449340028) q[19];
rz(1.2117923445499408) q[19];
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
ry(-3.065418100443857) q[0];
rz(0.526151926348386) q[0];
ry(-3.1358726199305984) q[1];
rz(0.2364363908108764) q[1];
ry(-2.967768830368963) q[2];
rz(-0.9930134032428866) q[2];
ry(-0.08984671490045049) q[3];
rz(-2.732851576939342) q[3];
ry(2.0945824157615647) q[4];
rz(-1.8291075391909304) q[4];
ry(-1.9817142479092216) q[5];
rz(1.2575778887982239) q[5];
ry(2.2682858695393744) q[6];
rz(0.11606165931281594) q[6];
ry(2.163996122077803) q[7];
rz(-0.2175101511701207) q[7];
ry(-2.9451771177710366) q[8];
rz(1.7720657241032038) q[8];
ry(0.03017594675371082) q[9];
rz(-3.00480466739027) q[9];
ry(1.724676204867585) q[10];
rz(-1.5777312431364896) q[10];
ry(-1.5253495332716787) q[11];
rz(-1.2701498619686582) q[11];
ry(0.3928152442753366) q[12];
rz(0.012452244765010237) q[12];
ry(-2.7775943078922567) q[13];
rz(0.7182915035448003) q[13];
ry(2.504383388322949) q[14];
rz(1.9210766173047555) q[14];
ry(2.48255524918348) q[15];
rz(2.0008612686241483) q[15];
ry(1.5235100908732147) q[16];
rz(-2.5056132626953147) q[16];
ry(-0.36012709367849993) q[17];
rz(2.1156198509156017) q[17];
ry(0.016541822640813383) q[18];
rz(2.5565925440800075) q[18];
ry(0.015454064916371801) q[19];
rz(1.3250601026772744) q[19];