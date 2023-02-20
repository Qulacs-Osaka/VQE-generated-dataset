OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-0.6197317857544293) q[0];
rz(-2.7034395462691885) q[0];
ry(0.26648846160540857) q[1];
rz(-1.4390112071818797) q[1];
ry(3.1131771724534243) q[2];
rz(-0.0946454863730653) q[2];
ry(-3.011673117878212) q[3];
rz(-2.61876033886656) q[3];
ry(-0.02369858674077907) q[4];
rz(2.356547764407514) q[4];
ry(1.6031330033209974) q[5];
rz(-1.9979479456761553) q[5];
ry(1.571768386924226) q[6];
rz(0.11067305584959401) q[6];
ry(-1.5705597897557595) q[7];
rz(0.5348717424570082) q[7];
ry(1.526081842092511) q[8];
rz(0.001108922776038135) q[8];
ry(2.570195095349589) q[9];
rz(-1.5700192119988854) q[9];
ry(1.6042415466312072) q[10];
rz(0.5095357797923744) q[10];
ry(3.141490722197829) q[11];
rz(1.981374083593876) q[11];
ry(-0.4429167347841112) q[12];
rz(-1.9113911163669144) q[12];
ry(1.941774796723914) q[13];
rz(1.5638167485715384) q[13];
ry(-3.0784300743745363) q[14];
rz(-0.7280156844913064) q[14];
ry(-0.3735586085215621) q[15];
rz(2.6154220693683703) q[15];
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
ry(1.9725975761616672) q[0];
rz(-2.752167282426828) q[0];
ry(0.33230075358005706) q[1];
rz(-1.895329465803986) q[1];
ry(3.1387182170709753) q[2];
rz(2.808516018380775) q[2];
ry(-3.0798445101902585) q[3];
rz(2.343065863987031) q[3];
ry(-1.178245919654988) q[4];
rz(1.6997688670270499) q[4];
ry(1.4182927679812094) q[5];
rz(1.3438656606973804) q[5];
ry(2.979128921084716) q[6];
rz(-2.439872936140984) q[6];
ry(2.4356005754880687) q[7];
rz(-2.6530676106812936) q[7];
ry(1.572325811155726) q[8];
rz(2.79828679333097) q[8];
ry(3.091969212279679) q[9];
rz(0.05418429267401735) q[9];
ry(1.545504739183435) q[10];
rz(1.6242440346530431) q[10];
ry(-0.0035573021078296506) q[11];
rz(2.407870388144377) q[11];
ry(3.080212365326783) q[12];
rz(1.6214610731547545) q[12];
ry(2.7787727785713883) q[13];
rz(-0.6944246056847365) q[13];
ry(0.7403782451379239) q[14];
rz(2.0393911276866383) q[14];
ry(2.1765454509883773) q[15];
rz(2.08959094467854) q[15];
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
ry(1.0874395794288438) q[0];
rz(1.0432171128858505) q[0];
ry(1.8876649129924579) q[1];
rz(0.2270455771052893) q[1];
ry(0.03058210752049373) q[2];
rz(0.868478248182919) q[2];
ry(0.7704541605230969) q[3];
rz(-0.7075687914852757) q[3];
ry(0.09705186232240237) q[4];
rz(-1.0800739488922695) q[4];
ry(0.0037230665381047245) q[5];
rz(1.0110315431874954) q[5];
ry(-0.00010232973970799209) q[6];
rz(0.9639576236078432) q[6];
ry(1.213633885634982) q[7];
rz(1.8012649186004746) q[7];
ry(1.545230104936053) q[8];
rz(-0.9993605259763996) q[8];
ry(-1.5712563496013827) q[9];
rz(2.1238547840818334) q[9];
ry(-1.3843816732867484) q[10];
rz(-0.03656953167878907) q[10];
ry(2.416658310811491) q[11];
rz(3.137529614074552) q[11];
ry(-2.701830819368024) q[12];
rz(1.8147839143456763) q[12];
ry(0.5663435513377724) q[13];
rz(-0.8784995201780716) q[13];
ry(1.4351656564572757) q[14];
rz(-2.494556111676421) q[14];
ry(-0.03060270548649413) q[15];
rz(1.791882802957094) q[15];
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
ry(1.068010275384367) q[0];
rz(2.259887194637898) q[0];
ry(-2.5665946269274613) q[1];
rz(-2.7557107530710434) q[1];
ry(3.113897018421709) q[2];
rz(-0.09917149130964731) q[2];
ry(2.7295494625249486) q[3];
rz(0.3253946636283592) q[3];
ry(2.3416288514297356) q[4];
rz(-1.4354424673111055) q[4];
ry(2.310136023011718) q[5];
rz(-2.860652106902435) q[5];
ry(0.2643751283943625) q[6];
rz(-2.950502422341566) q[6];
ry(-1.225126796686034) q[7];
rz(-2.0086043428884572) q[7];
ry(-2.471984607705701) q[8];
rz(1.6686383383779109) q[8];
ry(0.36212935096448984) q[9];
rz(1.1088780765402435) q[9];
ry(1.5714755099355573) q[10];
rz(-1.1065875119452109) q[10];
ry(0.36686788332007053) q[11];
rz(-3.1338819992278033) q[11];
ry(3.141395839899357) q[12];
rz(0.45917113770883) q[12];
ry(-0.4740275428731371) q[13];
rz(-2.178996540975851) q[13];
ry(3.0121159178795054) q[14];
rz(2.3980554719006184) q[14];
ry(1.1738839198516207) q[15];
rz(0.555310850031314) q[15];
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
ry(-1.6006363342162249) q[0];
rz(2.3177811946561775) q[0];
ry(-0.18310414183682777) q[1];
rz(-1.493871050146784) q[1];
ry(-3.1203304178292917) q[2];
rz(-1.8428551462117175) q[2];
ry(1.84511214559579) q[3];
rz(2.133342774612986) q[3];
ry(-2.132250563778684) q[4];
rz(0.65406093800415) q[4];
ry(-2.410127801382736) q[5];
rz(3.12976822053164) q[5];
ry(-0.0012619748189468538) q[6];
rz(-2.1818072743734627) q[6];
ry(-2.0246284001675705) q[7];
rz(-1.5948629379553925) q[7];
ry(0.8367549546905675) q[8];
rz(1.0557942504529416) q[8];
ry(-2.2038255944561085) q[9];
rz(-0.2414454085106156) q[9];
ry(2.4180796224625962) q[10];
rz(0.9813595357206754) q[10];
ry(1.5703769622420216) q[11];
rz(-2.9247521886144185) q[11];
ry(3.1278468201571274) q[12];
rz(-2.9455400810438266) q[12];
ry(-0.2566409968006047) q[13];
rz(1.0778734219595094) q[13];
ry(-1.904002868991675) q[14];
rz(2.897533319171508) q[14];
ry(1.1781204957754206) q[15];
rz(1.1900220386743339) q[15];
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
ry(0.49033416307868366) q[0];
rz(0.44882239892097) q[0];
ry(-2.7467479261124628) q[1];
rz(0.8824263845335628) q[1];
ry(3.1229786700157485) q[2];
rz(-2.5935725447878077) q[2];
ry(3.0948681452458424) q[3];
rz(2.5969622608932803) q[3];
ry(-3.098802608564386) q[4];
rz(-0.5894966213436557) q[4];
ry(-2.483983214042755) q[5];
rz(2.122287039571297) q[5];
ry(0.08971004837095276) q[6];
rz(1.00129018555045) q[6];
ry(-1.6627955492441429) q[7];
rz(-2.7494847853482276) q[7];
ry(-2.3663310376669977) q[8];
rz(-2.983961398363437) q[8];
ry(1.5783264365749017) q[9];
rz(0.942601617041051) q[9];
ry(-0.5033521398851996) q[10];
rz(1.4040121944102957) q[10];
ry(-0.17697511511860398) q[11];
rz(2.033627603079715) q[11];
ry(1.543021299826851) q[12];
rz(-0.9310739785839157) q[12];
ry(2.4111122353499437) q[13];
rz(0.23731556463507353) q[13];
ry(0.8211422145026779) q[14];
rz(2.923951750074082) q[14];
ry(2.8478960758128964) q[15];
rz(0.9282545476361473) q[15];
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
ry(-2.166223657039132) q[0];
rz(1.9559560919154535) q[0];
ry(1.8792153184748575) q[1];
rz(0.07412376245160957) q[1];
ry(-2.7818542058278766) q[2];
rz(0.35944443946073895) q[2];
ry(0.741611362393809) q[3];
rz(-2.430384929661282) q[3];
ry(-1.9299404398269167) q[4];
rz(-2.229389423218891) q[4];
ry(1.2378011447297474) q[5];
rz(-2.8124461944143455) q[5];
ry(2.791866137691833) q[6];
rz(1.5419306048751684) q[6];
ry(-0.7001935803970372) q[7];
rz(-0.8817113658729002) q[7];
ry(0.2994615977677091) q[8];
rz(-1.6183082058448859) q[8];
ry(-2.057769657388942) q[9];
rz(-2.7995757620392285) q[9];
ry(-1.9061089129982038) q[10];
rz(-1.6797251965180375) q[10];
ry(-3.089460935910283) q[11];
rz(2.8946685929374594) q[11];
ry(-0.005673847483594074) q[12];
rz(-1.9205095323358883) q[12];
ry(-1.566096615484275) q[13];
rz(2.1827134331382023) q[13];
ry(2.7549599285149746) q[14];
rz(1.056900114084839) q[14];
ry(0.4749719458064323) q[15];
rz(2.2508090869522595) q[15];
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
ry(1.8788320157785188) q[0];
rz(-2.938475962182265) q[0];
ry(0.03460922822064649) q[1];
rz(-0.16540497272545446) q[1];
ry(-2.950514910197005) q[2];
rz(0.03543820897644917) q[2];
ry(1.8277511766852241) q[3];
rz(0.02981982099954905) q[3];
ry(1.8509695265415447) q[4];
rz(0.44542914386924887) q[4];
ry(0.6409026793689021) q[5];
rz(0.45009035645699313) q[5];
ry(2.5147159882860146) q[6];
rz(-2.9842105809263404) q[6];
ry(1.747160339819973) q[7];
rz(0.2694030078933433) q[7];
ry(-0.03001693679677595) q[8];
rz(-0.8074378203205246) q[8];
ry(0.9186785947832741) q[9];
rz(3.1041414178761895) q[9];
ry(-2.70113160596669) q[10];
rz(2.519118192912444) q[10];
ry(1.6226716985552712) q[11];
rz(-1.767682078379058) q[11];
ry(0.6951522644927015) q[12];
rz(1.661749937298187) q[12];
ry(2.2516934185338817) q[13];
rz(2.0949213307721846) q[13];
ry(1.9413266391268946) q[14];
rz(2.7319946841768488) q[14];
ry(2.1655752162880466) q[15];
rz(2.555852896083397) q[15];
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
ry(2.7306502188119564) q[0];
rz(-2.1427309733186783) q[0];
ry(0.6267460298165249) q[1];
rz(0.7403052556561797) q[1];
ry(-0.43638811553959217) q[2];
rz(-2.550380565021863) q[2];
ry(3.1099038822974916) q[3];
rz(2.139803155552336) q[3];
ry(-0.011949684879176914) q[4];
rz(1.8819525253737945) q[4];
ry(0.8787151304121859) q[5];
rz(-1.4879077741177298) q[5];
ry(-1.2423563484970215) q[6];
rz(2.1704474628853134) q[6];
ry(-2.9836360296023314) q[7];
rz(-1.1876592479123227) q[7];
ry(0.684914283450615) q[8];
rz(-0.0010004805536724158) q[8];
ry(1.5238623851478508) q[9];
rz(-1.2017237850697784) q[9];
ry(-0.8453038063899332) q[10];
rz(-1.855004504979938) q[10];
ry(0.4421440789515204) q[11];
rz(-0.8924428048844187) q[11];
ry(3.1103351123633036) q[12];
rz(0.5911500120707921) q[12];
ry(-3.0564173393746334) q[13];
rz(1.1758427801337543) q[13];
ry(-2.5325485705876614) q[14];
rz(-0.7588980455149829) q[14];
ry(1.9118538289206946) q[15];
rz(-0.12888095004586117) q[15];
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
ry(1.6421724027445093) q[0];
rz(-2.64455429094537) q[0];
ry(-1.4989894620000643) q[1];
rz(1.2560102130532882) q[1];
ry(-2.92221809085202) q[2];
rz(-2.6105518335331537) q[2];
ry(-0.14728914017030534) q[3];
rz(2.9923849569390204) q[3];
ry(1.8436519991134905) q[4];
rz(0.7910261818568429) q[4];
ry(-2.57492849480091) q[5];
rz(1.0294811660255265) q[5];
ry(-3.0242064903476606) q[6];
rz(-0.3345462434901645) q[6];
ry(2.8131748337659657) q[7];
rz(3.137190649388785) q[7];
ry(-2.936778584882388) q[8];
rz(0.013862631079685883) q[8];
ry(0.32137261527355104) q[9];
rz(-2.141804721496163) q[9];
ry(-1.279266196382204) q[10];
rz(-0.8010391783074757) q[10];
ry(-1.7605385045033861) q[11];
rz(2.083299595905915) q[11];
ry(-0.4049639586161457) q[12];
rz(1.498532390698239) q[12];
ry(-0.7963050885224585) q[13];
rz(1.195825684014437) q[13];
ry(1.1255637483070644) q[14];
rz(-3.0366707866277536) q[14];
ry(-0.10381338311359051) q[15];
rz(-0.9825078287133681) q[15];
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
ry(2.861292048388833) q[0];
rz(2.9669591767655663) q[0];
ry(0.7579717731091007) q[1];
rz(-3.0794001719010575) q[1];
ry(2.003647265821784) q[2];
rz(1.273910247768797) q[2];
ry(-3.1283187224760085) q[3];
rz(0.4589184405016508) q[3];
ry(3.141443349909828) q[4];
rz(-2.5632226936954607) q[4];
ry(-1.2157234767461818) q[5];
rz(-2.38783708330774) q[5];
ry(1.610486304426324) q[6];
rz(-1.4867043630926347) q[6];
ry(0.15824873320686184) q[7];
rz(3.135503633059811) q[7];
ry(0.5193461244001822) q[8];
rz(2.3774335557320363) q[8];
ry(2.268299136708923) q[9];
rz(3.0292412489085003) q[9];
ry(-0.04550807008837872) q[10];
rz(0.8199491954741847) q[10];
ry(-0.030093132282253385) q[11];
rz(2.2934674047901065) q[11];
ry(-3.1167383931023425) q[12];
rz(-1.4020658237674601) q[12];
ry(1.231457406231769) q[13];
rz(-2.138381534292582) q[13];
ry(0.6228797007839058) q[14];
rz(2.21490999241626) q[14];
ry(0.892329584048279) q[15];
rz(0.3608816688791568) q[15];
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
ry(-0.20936593593901964) q[0];
rz(-2.220703930537227) q[0];
ry(2.706372211619133) q[1];
rz(2.681350991308521) q[1];
ry(2.940585275919689) q[2];
rz(-1.695864896119332) q[2];
ry(-1.2927872800321012) q[3];
rz(-0.42312737748451634) q[3];
ry(2.7800973767918666) q[4];
rz(-1.4808786993612555) q[4];
ry(2.8926130185530683) q[5];
rz(1.9685683275952015) q[5];
ry(1.1868329941882785) q[6];
rz(-0.7064742985169739) q[6];
ry(2.2084822537323436) q[7];
rz(1.2002543372647352) q[7];
ry(-0.03723099443465827) q[8];
rz(0.5295485960356867) q[8];
ry(0.202961551616854) q[9];
rz(-2.9828008823726537) q[9];
ry(1.1668445375975598) q[10];
rz(-0.28684042298288936) q[10];
ry(-2.521013068843671) q[11];
rz(-1.0260773211435914) q[11];
ry(3.09574502413208) q[12];
rz(-1.4879034313987374) q[12];
ry(-2.8408278607661157) q[13];
rz(-1.561779473070116) q[13];
ry(1.513586276143437) q[14];
rz(2.6942559812851457) q[14];
ry(-1.4897268949650169) q[15];
rz(2.0777086445461705) q[15];
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
ry(-1.6476825923614715) q[0];
rz(2.4574985882398335) q[0];
ry(0.8403928239963356) q[1];
rz(2.113620557919344) q[1];
ry(-2.167543633443363) q[2];
rz(0.05110386590027164) q[2];
ry(-3.136712854017306) q[3];
rz(2.851000407144163) q[3];
ry(3.1408956222660622) q[4];
rz(-0.14501358826740376) q[4];
ry(1.864958395477341) q[5];
rz(-2.1239553046345723) q[5];
ry(-0.7554758472790963) q[6];
rz(-0.43406874803230355) q[6];
ry(-3.130825952892849) q[7];
rz(1.4802863837978224) q[7];
ry(1.1374452673782134) q[8];
rz(-2.8613705613735045) q[8];
ry(-2.225148980188939) q[9];
rz(1.8701974004185322) q[9];
ry(-0.025572451144786807) q[10];
rz(-1.623549412710389) q[10];
ry(-3.1173201547471336) q[11];
rz(3.1357611809359756) q[11];
ry(1.5740793832148519) q[12];
rz(-2.2267269949948307) q[12];
ry(-3.1069828074130297) q[13];
rz(1.6836726799247488) q[13];
ry(3.0051940538194373) q[14];
rz(1.1724078375988707) q[14];
ry(1.4846219480552785) q[15];
rz(0.8843122792499046) q[15];
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
ry(-1.9651902881291092) q[0];
rz(-1.2920957741621937) q[0];
ry(2.9380382230797943) q[1];
rz(1.2883654409712664) q[1];
ry(-3.0559922647668976) q[2];
rz(-2.3514255002354703) q[2];
ry(-2.2803524378278963) q[3];
rz(-1.5289693542025715) q[3];
ry(1.2700529262250793) q[4];
rz(-3.0494290702846665) q[4];
ry(1.339490974965173) q[5];
rz(-1.2459181105176782) q[5];
ry(-2.423403988407723) q[6];
rz(-1.1983748282100128) q[6];
ry(-3.02675434403449) q[7];
rz(-0.3300649659481296) q[7];
ry(2.2612286141824116) q[8];
rz(0.8469647790696435) q[8];
ry(1.5434168557299344) q[9];
rz(1.1095435871525041) q[9];
ry(1.0167301852207213) q[10];
rz(-2.557134939168782) q[10];
ry(0.024176072257689363) q[11];
rz(0.14588345863530705) q[11];
ry(-0.04073357668092381) q[12];
rz(1.9684331973648754) q[12];
ry(1.5284042543541854) q[13];
rz(2.6953838236126844) q[13];
ry(1.5741852391053073) q[14];
rz(-2.7135530654563587) q[14];
ry(1.5841767221332252) q[15];
rz(1.5134146061018139) q[15];
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
ry(-2.305068277793397) q[0];
rz(0.25922118083846934) q[0];
ry(-3.0650075581725797) q[1];
rz(-0.3605221676936647) q[1];
ry(2.4278796677371326) q[2];
rz(2.314576148257737) q[2];
ry(-3.138922049417192) q[3];
rz(-1.112409175553509) q[3];
ry(3.133215156115879) q[4];
rz(0.3004793460942751) q[4];
ry(0.408727913244193) q[5];
rz(2.5464161167098456) q[5];
ry(-1.3363835256044254) q[6];
rz(2.6555551350972926) q[6];
ry(-0.9129803614314929) q[7];
rz(-1.267003765380804) q[7];
ry(-0.03143268008811262) q[8];
rz(-2.201027470104223) q[8];
ry(-0.03805641468414096) q[9];
rz(-2.550758584762439) q[9];
ry(-1.6229186675261955) q[10];
rz(-0.005976373979618312) q[10];
ry(-0.0036529802094849018) q[11];
rz(-1.4553330155641202) q[11];
ry(-3.1172084290141946) q[12];
rz(-0.3443385454155144) q[12];
ry(-2.0002129823490944) q[13];
rz(-1.205673534354029) q[13];
ry(2.5579926747783905) q[14];
rz(2.7747787142200835) q[14];
ry(-1.629749381394141) q[15];
rz(1.6748462502215944) q[15];
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
ry(-2.987272923325421) q[0];
rz(-2.4844421688639184) q[0];
ry(-1.582965037651301) q[1];
rz(-0.9815989221490831) q[1];
ry(-2.248711418831003) q[2];
rz(-0.7198416445906154) q[2];
ry(0.45172564102204404) q[3];
rz(1.3576137384828464) q[3];
ry(-1.4178005373111586) q[4];
rz(-0.07813701223123083) q[4];
ry(-1.2965673643239173) q[5];
rz(0.13993886742478612) q[5];
ry(0.7470993919857243) q[6];
rz(1.369697793639292) q[6];
ry(2.0041705402469416) q[7];
rz(-2.4192254675451474) q[7];
ry(3.1054990252012935) q[8];
rz(-1.0729415940224465) q[8];
ry(1.6325525908182428) q[9];
rz(-1.7579159830644802) q[9];
ry(-0.7777504010758154) q[10];
rz(-0.39433098414818174) q[10];
ry(-2.0239758046726735) q[11];
rz(3.0460510654103543) q[11];
ry(0.11642903392945046) q[12];
rz(-2.0973628384930647) q[12];
ry(3.126208477526419) q[13];
rz(-1.5059195782084434) q[13];
ry(-0.08150042869721297) q[14];
rz(-2.0790018966630464) q[14];
ry(-2.449128646974608) q[15];
rz(-1.8128777657791444) q[15];
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
ry(1.5380055186796715) q[0];
rz(1.3522608115481853) q[0];
ry(2.3084285816554964) q[1];
rz(-0.061312432045636765) q[1];
ry(0.6555184225181674) q[2];
rz(-1.8623212186006262) q[2];
ry(0.6556297751357162) q[3];
rz(-0.5868488024994728) q[3];
ry(-3.1250645423700036) q[4];
rz(-0.9305358660884426) q[4];
ry(-3.0861918891995654) q[5];
rz(2.1577715503012094) q[5];
ry(0.3122554761077108) q[6];
rz(2.068095130131627) q[6];
ry(1.7840173937289219) q[7];
rz(1.4866334433297113) q[7];
ry(1.6840233905534037) q[8];
rz(2.237467174496673) q[8];
ry(1.2186074405951475) q[9];
rz(1.6099797878872042) q[9];
ry(-0.007664599175612351) q[10];
rz(-1.4211918600434383) q[10];
ry(1.6420224848524931) q[11];
rz(-0.08205591576573232) q[11];
ry(-0.0005260416940703207) q[12];
rz(-0.8073993996977302) q[12];
ry(-1.0779047774019446) q[13];
rz(1.5716849110808815) q[13];
ry(2.0650673227090985) q[14];
rz(-1.250338918931134) q[14];
ry(-0.0052910225806254456) q[15];
rz(-0.6550511521572702) q[15];
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
ry(0.3804647141277009) q[0];
rz(1.5058727936990615) q[0];
ry(0.927406824924474) q[1];
rz(-2.2719958362662354) q[1];
ry(-2.069121482309572) q[2];
rz(1.366952169748857) q[2];
ry(2.5636220761877553) q[3];
rz(3.060031454971383) q[3];
ry(3.1408131215162847) q[4];
rz(3.127215023800917) q[4];
ry(1.4820225305027135) q[5];
rz(-1.1923950707289181) q[5];
ry(-2.7361635789265044) q[6];
rz(-1.854666876363746) q[6];
ry(2.065661773269742) q[7];
rz(-2.133852835509095) q[7];
ry(-0.012444243646225261) q[8];
rz(-2.2327877674798158) q[8];
ry(0.006255131690565412) q[9];
rz(-0.7303596752117079) q[9];
ry(-3.1177148979837868) q[10];
rz(-2.146328439602478) q[10];
ry(-0.4632860109001893) q[11];
rz(-0.8498203568396238) q[11];
ry(3.039870378217723) q[12];
rz(3.113127477509097) q[12];
ry(0.045269390285080896) q[13];
rz(0.3764780022553601) q[13];
ry(-1.533568019013114) q[14];
rz(-2.8154376641753354) q[14];
ry(3.0738933038986302) q[15];
rz(1.5736080965188837) q[15];
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
ry(-2.6923737785582342) q[0];
rz(0.4323994432694373) q[0];
ry(2.0663494812503007) q[1];
rz(2.433378604459241) q[1];
ry(0.18718563530976323) q[2];
rz(1.7115716027096772) q[2];
ry(-2.6762576868021903) q[3];
rz(2.6961049242828192) q[3];
ry(2.7600738316751103) q[4];
rz(0.8785234634254513) q[4];
ry(1.5879030356608697) q[5];
rz(1.546959597886728) q[5];
ry(0.15003493486275318) q[6];
rz(-3.0601464022339138) q[6];
ry(-0.11190391933511133) q[7];
rz(-0.9969220851926002) q[7];
ry(0.8765376178755688) q[8];
rz(1.4651670374398924) q[8];
ry(1.9144713944547114) q[9];
rz(-2.2051416015238177) q[9];
ry(-3.1353499963869473) q[10];
rz(-1.7441230350828718) q[10];
ry(3.0009321522085597) q[11];
rz(0.3030906113608483) q[11];
ry(1.5554982037338754) q[12];
rz(-0.05674658381946546) q[12];
ry(1.2819901889771503) q[13];
rz(-1.9070425247312341) q[13];
ry(0.7588417824957325) q[14];
rz(-1.0416469661588967) q[14];
ry(-1.1239870595516541) q[15];
rz(-2.7108350021352567) q[15];
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
ry(0.43445364039913653) q[0];
rz(0.2763900087774753) q[0];
ry(-0.2063115746014594) q[1];
rz(2.6739613028078826) q[1];
ry(1.1319146609999118) q[2];
rz(-2.474609984037041) q[2];
ry(1.4977452865430205) q[3];
rz(-0.032058664576044026) q[3];
ry(-0.041557654677645633) q[4];
rz(-2.4311099365675095) q[4];
ry(-1.3444769986949725) q[5];
rz(-1.233098708173272) q[5];
ry(-0.23961276217685068) q[6];
rz(-2.3935482738716227) q[6];
ry(1.9695867982785131) q[7];
rz(2.1863631387972835) q[7];
ry(-0.002667564878263917) q[8];
rz(2.2183137290189796) q[8];
ry(-0.0015477992493620938) q[9];
rz(-2.807360080999514) q[9];
ry(0.013998248099682486) q[10];
rz(2.864883814249893) q[10];
ry(1.4093062889327759) q[11];
rz(1.1466014409239502) q[11];
ry(-0.005276879525463427) q[12];
rz(2.0352364892960573) q[12];
ry(-0.6523956765490907) q[13];
rz(3.124222681588681) q[13];
ry(-3.122232762457475) q[14];
rz(1.3122534824000482) q[14];
ry(3.116114569584513) q[15];
rz(0.4317392017720093) q[15];
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
ry(-2.006019295892797) q[0];
rz(-3.0543423876797906) q[0];
ry(1.6817411064416485) q[1];
rz(2.816923320226625) q[1];
ry(0.09208932152193967) q[2];
rz(1.3631838565328351) q[2];
ry(-0.1655291417056511) q[3];
rz(1.473544740802721) q[3];
ry(0.2645055044432336) q[4];
rz(-3.05763208348764) q[4];
ry(-3.108879218883822) q[5];
rz(0.3178182013000282) q[5];
ry(-3.1304635361872375) q[6];
rz(0.660507019102522) q[6];
ry(1.9165066587218649) q[7];
rz(0.9951972781561733) q[7];
ry(0.8039958062022357) q[8];
rz(0.08173996394523123) q[8];
ry(2.0748899547517907) q[9];
rz(-1.1513762146074784) q[9];
ry(0.13996506415197096) q[10];
rz(-1.114202077552525) q[10];
ry(1.6116775499365348) q[11];
rz(-0.04392015409698496) q[11];
ry(-2.990402418481601) q[12];
rz(-2.787566771422891) q[12];
ry(-0.36248326694664623) q[13];
rz(0.027355075626561035) q[13];
ry(0.009187049302388672) q[14];
rz(2.4365910413688727) q[14];
ry(1.1149954530932367) q[15];
rz(1.6707702314171575) q[15];
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
ry(2.2015216807500897) q[0];
rz(-0.009382837384234398) q[0];
ry(-0.036938353285779754) q[1];
rz(-0.215312937479808) q[1];
ry(-1.539697520129823) q[2];
rz(1.4928541752968394) q[2];
ry(1.5629232503978494) q[3];
rz(-1.4799789470824138) q[3];
ry(-1.730179890564302) q[4];
rz(-2.753916820803015) q[4];
ry(-1.467336366134205) q[5];
rz(2.4971817545493193) q[5];
ry(-1.1906866040158441) q[6];
rz(1.6877900056059747) q[6];
ry(0.36299729251115354) q[7];
rz(2.970646199905829) q[7];
ry(-0.1950254814666259) q[8];
rz(-1.3189946954623162) q[8];
ry(0.009653849348420293) q[9];
rz(2.541242338173535) q[9];
ry(0.013890484208136203) q[10];
rz(-2.14419968329958) q[10];
ry(2.7430932367053718) q[11];
rz(-2.503440186537955) q[11];
ry(2.253698476363212) q[12];
rz(-2.2082491626351537) q[12];
ry(-2.4711123742221384) q[13];
rz(2.6224821408898484) q[13];
ry(-1.5019875110281473) q[14];
rz(0.09670208086010314) q[14];
ry(1.8359059724446176) q[15];
rz(-2.06337157155391) q[15];
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
ry(-1.7092933871049842) q[0];
rz(-2.772002464867173) q[0];
ry(1.996821430588616) q[1];
rz(-1.596581923416631) q[1];
ry(0.01861842281101822) q[2];
rz(0.7734551950458092) q[2];
ry(-2.470401196496086) q[3];
rz(0.017813485984187517) q[3];
ry(0.0044950629903737875) q[4];
rz(-3.045112810311301) q[4];
ry(0.025334616654563824) q[5];
rz(0.7803943236155503) q[5];
ry(3.126339806856361) q[6];
rz(-2.6611800441295834) q[6];
ry(0.6360868860425103) q[7];
rz(-1.4240676839921385) q[7];
ry(0.7235830413700857) q[8];
rz(1.159329248089633) q[8];
ry(-0.3139040198181625) q[9];
rz(1.6039809952279525) q[9];
ry(-0.14623110135611483) q[10];
rz(-0.11760824511356877) q[10];
ry(-3.1301346870163242) q[11];
rz(-2.7244045284378) q[11];
ry(-1.6849685509281) q[12];
rz(1.6203515961554125) q[12];
ry(-3.004321098474593) q[13];
rz(-1.78315015548581) q[13];
ry(1.6981308045155057) q[14];
rz(2.317810952289794) q[14];
ry(-3.125805627723992) q[15];
rz(-1.8533184242479646) q[15];
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
ry(1.8570480207442832) q[0];
rz(1.2024277768756966) q[0];
ry(1.5457717693129833) q[1];
rz(2.0502333023546164) q[1];
ry(-0.008037819449819926) q[2];
rz(-0.09438789308131376) q[2];
ry(3.0243483726037477) q[3];
rz(-0.01745864579997214) q[3];
ry(2.941935527640137) q[4];
rz(1.9930602203223031) q[4];
ry(-0.10758456209945116) q[5];
rz(1.700314766007494) q[5];
ry(-0.982400449773654) q[6];
rz(-2.538481478609477) q[6];
ry(0.3565428591528983) q[7];
rz(-0.34042218150184894) q[7];
ry(2.1254940640445343) q[8];
rz(1.4265788413028593) q[8];
ry(0.17758674351912876) q[9];
rz(-1.3858767314950478) q[9];
ry(0.0016554664682525964) q[10];
rz(2.80721487051856) q[10];
ry(0.033272396312979256) q[11];
rz(1.6636943362078112) q[11];
ry(-1.586382051326483) q[12];
rz(-2.999023417469022) q[12];
ry(-2.4660524947125997) q[13];
rz(-0.03525643811738238) q[13];
ry(-0.9967187987434308) q[14];
rz(-0.958268674145879) q[14];
ry(2.897846281251899) q[15];
rz(1.3213181401291818) q[15];
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
ry(-0.00930621052019807) q[0];
rz(1.0209628499548966) q[0];
ry(-2.7015374973866386) q[1];
rz(0.9182800595511882) q[1];
ry(-1.3744374956692471) q[2];
rz(1.565633582885557) q[2];
ry(2.2459629108846455) q[3];
rz(3.139628830188748) q[3];
ry(-2.7611994062101957) q[4];
rz(1.5267574877962549) q[4];
ry(3.134152347708856) q[5];
rz(-0.6022029567846733) q[5];
ry(3.122891863946873) q[6];
rz(1.3360119651511948) q[6];
ry(3.1331763432344997) q[7];
rz(-1.9469199933081613) q[7];
ry(2.8446260027035426) q[8];
rz(0.22496895407784953) q[8];
ry(3.075874767897153) q[9];
rz(-2.774676040125173) q[9];
ry(-0.028188248095010117) q[10];
rz(-2.119404021568916) q[10];
ry(2.7112817145472867) q[11];
rz(2.6554095565112807) q[11];
ry(2.436321789327998) q[12];
rz(-3.0783807655017608) q[12];
ry(-1.049639074601143) q[13];
rz(3.0686560931809708) q[13];
ry(1.609288605778418) q[14];
rz(2.86651915530319) q[14];
ry(0.027618749920119253) q[15];
rz(1.8695739630505686) q[15];
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
ry(0.35274194963024935) q[0];
rz(-2.450970158240526) q[0];
ry(-1.8821769548649374) q[1];
rz(-0.010593722981766085) q[1];
ry(1.5715815944532974) q[2];
rz(-0.17470612455407167) q[2];
ry(-3.115125741849116) q[3];
rz(-3.1154080009663447) q[3];
ry(1.6651629013316203) q[4];
rz(-1.9522943204327496) q[4];
ry(0.031041594283299955) q[5];
rz(-0.8597954259910575) q[5];
ry(3.0919538702981333) q[6];
rz(-0.3691189640426371) q[6];
ry(-1.4608688461467483) q[7];
rz(-0.7953620228396715) q[7];
ry(-1.9751920209282652) q[8];
rz(-0.8340076570011989) q[8];
ry(0.37620502095708097) q[9];
rz(0.9122939026545317) q[9];
ry(0.8388315985477451) q[10];
rz(0.1105575370616556) q[10];
ry(1.5886816252522773) q[11];
rz(-1.6098568428673732) q[11];
ry(-0.07816353819580851) q[12];
rz(1.3560172776659796) q[12];
ry(1.5797022177420879) q[13];
rz(-3.1363191859031043) q[13];
ry(-0.005345951084877641) q[14];
rz(-2.8314793384295602) q[14];
ry(2.550268156266783) q[15];
rz(-1.3853888137158634) q[15];
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
ry(0.0015579360722890456) q[0];
rz(0.05736331467860352) q[0];
ry(-1.5804291657604352) q[1];
rz(-3.133595486464736) q[1];
ry(2.9087177172787615) q[2];
rz(-1.7521385692922289) q[2];
ry(-1.556156877611791) q[3];
rz(1.487641427339842) q[3];
ry(-0.03996269532012007) q[4];
rz(-1.2705442601325307) q[4];
ry(0.00360175073440536) q[5];
rz(-0.6112695914588891) q[5];
ry(0.005936320759341075) q[6];
rz(-1.6575766128091747) q[6];
ry(0.2987849156584882) q[7];
rz(-2.829814356774752) q[7];
ry(1.788765122085374) q[8];
rz(-0.23885963754351644) q[8];
ry(3.1415663353032373) q[9];
rz(-2.1632686315595966) q[9];
ry(-3.083144661923558) q[10];
rz(-2.9928382219315344) q[10];
ry(-0.6815781684554566) q[11];
rz(-0.012197799002680476) q[11];
ry(-0.015446656820945035) q[12];
rz(-2.1964751272368668) q[12];
ry(-2.1377851178505543) q[13];
rz(-1.9725750119149303) q[13];
ry(-0.1522322911896179) q[14];
rz(-3.0627664061515887) q[14];
ry(3.1414637607599403) q[15];
rz(-1.3876189995215) q[15];
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
ry(3.107442359147069) q[0];
rz(0.11616828514897275) q[0];
ry(-2.8276152302469617) q[1];
rz(-0.12164684265367079) q[1];
ry(1.5605324369341056) q[2];
rz(-1.5494223368654136) q[2];
ry(3.140351162107267) q[3];
rz(1.4884099492794283) q[3];
ry(3.0722205048333633) q[4];
rz(2.3447750522836257) q[4];
ry(2.572039980179324) q[5];
rz(-2.9895956916637316) q[5];
ry(1.75515906125957) q[6];
rz(2.9589832222837984) q[6];
ry(-2.915093256447398) q[7];
rz(2.925989354351897) q[7];
ry(-0.9645514690343973) q[8];
rz(-0.028411501992191113) q[8];
ry(-3.100720351787603) q[9];
rz(-0.1956456237340738) q[9];
ry(-0.9917663183208899) q[10];
rz(0.31047669581251114) q[10];
ry(3.09011789608511) q[11];
rz(3.059943808256322) q[11];
ry(-3.140243774089734) q[12];
rz(0.7078015881373183) q[12];
ry(3.105172678645891) q[13];
rz(-0.8841408280329484) q[13];
ry(-2.327893914848444) q[14];
rz(1.7053531851654713) q[14];
ry(-2.607171161696654) q[15];
rz(2.8183277349961906) q[15];
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
ry(3.1391588433362068) q[0];
rz(-1.8378231094413466) q[0];
ry(-3.063943853961828) q[1];
rz(0.0006253342960685928) q[1];
ry(1.567948144772772) q[2];
rz(-1.5024006544951716) q[2];
ry(1.570123780862189) q[3];
rz(1.3826993787936588) q[3];
ry(-3.12864371815024) q[4];
rz(-0.7945071572893685) q[4];
ry(-3.1289485422999443) q[5];
rz(3.0264077013673556) q[5];
ry(-0.026708589376764458) q[6];
rz(2.9333799188597847) q[6];
ry(0.2757725983863093) q[7];
rz(-2.8834287415224655) q[7];
ry(-1.633096781302199) q[8];
rz(-3.0614561924934245) q[8];
ry(0.03704181763075453) q[9];
rz(2.152848635065488) q[9];
ry(-3.087414292555379) q[10];
rz(0.27911102010936517) q[10];
ry(-1.1643662036896503) q[11];
rz(-3.0361423492646233) q[11];
ry(-0.009783607779828096) q[12];
rz(1.8771191042629047) q[12];
ry(3.1342052779773577) q[13];
rz(2.2771764084340584) q[13];
ry(-1.4932961456291878) q[14];
rz(2.030565908578858) q[14];
ry(0.018767826022540606) q[15];
rz(1.5096292324789253) q[15];
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
ry(3.1361396023056045) q[0];
rz(1.616058482809784) q[0];
ry(-1.5680720524811718) q[1];
rz(2.0298222060856266) q[1];
ry(1.5893952880703681) q[2];
rz(-1.3917755059964652) q[2];
ry(-0.12966878914728497) q[3];
rz(0.43437898004923337) q[3];
ry(1.582510144410592) q[4];
rz(1.6254263070388335) q[4];
ry(-0.49499152258858364) q[5];
rz(-1.6686449970315016) q[5];
ry(0.7363757469411425) q[6];
rz(0.8789552294493569) q[6];
ry(-1.8105097372484973) q[7];
rz(-2.7357151338510746) q[7];
ry(1.6184525513897554) q[8];
rz(3.129690507285913) q[8];
ry(-2.9432055530850234) q[9];
rz(-0.26955859801400717) q[9];
ry(-0.6941307541520135) q[10];
rz(-0.8050704248574432) q[10];
ry(2.2214788770725646) q[11];
rz(-3.0986639322074443) q[11];
ry(1.2231763771161532) q[12];
rz(2.3230731038999117) q[12];
ry(-2.0132318362690986) q[13];
rz(2.6580159495980955) q[13];
ry(-1.8976368740090575) q[14];
rz(2.561769442680827) q[14];
ry(0.5689011064902542) q[15];
rz(0.29043901301790687) q[15];