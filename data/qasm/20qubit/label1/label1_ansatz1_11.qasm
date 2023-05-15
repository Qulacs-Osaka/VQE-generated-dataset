OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(0.857505849594852) q[0];
rz(-0.9532391782225238) q[0];
ry(-1.6767216985222173) q[1];
rz(-1.5483717328596618) q[1];
ry(-0.03602396848095268) q[2];
rz(-0.87652494891246) q[2];
ry(-2.8934771364842553) q[3];
rz(1.8810645326547029) q[3];
ry(-1.9505726768794454) q[4];
rz(-0.34685312978061633) q[4];
ry(-0.7324288263269327) q[5];
rz(0.08196113372421364) q[5];
ry(-1.0054652338897005) q[6];
rz(-1.7496578332122805) q[6];
ry(-0.0011877073710764904) q[7];
rz(-2.0188422297443314) q[7];
ry(1.4851885739883661) q[8];
rz(2.980441573819411) q[8];
ry(1.9243987323763605) q[9];
rz(0.33318288889387393) q[9];
ry(0.3130088663614413) q[10];
rz(-0.17784876779097586) q[10];
ry(2.7690068453406957) q[11];
rz(0.13636390193169223) q[11];
ry(-0.7951652064712152) q[12];
rz(0.20340320525663289) q[12];
ry(0.2484044884577159) q[13];
rz(-0.823370319963183) q[13];
ry(2.0511916751686883) q[14];
rz(0.48346403554295053) q[14];
ry(0.7263168774456117) q[15];
rz(-2.7493710491695134) q[15];
ry(-3.128009425605027) q[16];
rz(0.9237778897909319) q[16];
ry(-0.8404459521356529) q[17];
rz(1.8096326360793773) q[17];
ry(-0.05888192807498882) q[18];
rz(2.0311943085689013) q[18];
ry(-2.5859990818094025) q[19];
rz(2.0717169692491) q[19];
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
ry(0.1082807338782175) q[0];
rz(1.9691778312307298) q[0];
ry(0.6814027596187762) q[1];
rz(1.0809981528609545) q[1];
ry(-0.09280293958374575) q[2];
rz(-1.1299572113470875) q[2];
ry(0.18059327088230753) q[3];
rz(-0.7466331035519566) q[3];
ry(0.901873401758813) q[4];
rz(1.2379797467726226) q[4];
ry(2.8919440250301696) q[5];
rz(-1.9141282535342272) q[5];
ry(-0.28188326324650087) q[6];
rz(-1.444397536757756) q[6];
ry(0.4313270656631518) q[7];
rz(1.0785041105297388) q[7];
ry(1.9017162622969312) q[8];
rz(-1.69454153338152) q[8];
ry(-1.2766146055734588) q[9];
rz(1.3464042379560304) q[9];
ry(1.5495562212055543) q[10];
rz(-0.6087446093233908) q[10];
ry(-2.0101498976844514) q[11];
rz(3.069293727966754) q[11];
ry(-1.0788209522675523) q[12];
rz(-2.091966581824513) q[12];
ry(2.7871427919442113) q[13];
rz(0.8698552660198305) q[13];
ry(-3.0770179296008195) q[14];
rz(1.9297784648138787) q[14];
ry(-2.181448958779943) q[15];
rz(2.2911077540089813) q[15];
ry(3.1023954507693654) q[16];
rz(-1.2628948655830792) q[16];
ry(-1.285995775288954) q[17];
rz(0.9847567014790304) q[17];
ry(-0.08985548109193242) q[18];
rz(0.9673004638468266) q[18];
ry(-2.139398180506535) q[19];
rz(-2.4094498116399032) q[19];
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
ry(1.8417084952771605) q[0];
rz(-2.5143071755203343) q[0];
ry(1.6787766729201836) q[1];
rz(-1.3056290326313849) q[1];
ry(3.0988398684269707) q[2];
rz(-1.4021164820305279) q[2];
ry(3.0413222247863794) q[3];
rz(-1.376103489261507) q[3];
ry(-2.7042796165318252) q[4];
rz(1.390434275020457) q[4];
ry(-0.6824395011138841) q[5];
rz(-1.285674087987089) q[5];
ry(-2.08291668529682) q[6];
rz(-2.9339518356358454) q[6];
ry(1.64414864401399) q[7];
rz(2.3228167259760983) q[7];
ry(-2.505583600414286) q[8];
rz(2.4895995386081884) q[8];
ry(2.3789299663117034) q[9];
rz(1.2607149526956538) q[9];
ry(-1.907263521820151) q[10];
rz(2.8546970026967684) q[10];
ry(-1.557718551719585) q[11];
rz(3.1230408159394556) q[11];
ry(0.7959125175518269) q[12];
rz(-2.5861673116325465) q[12];
ry(2.2820138589115553) q[13];
rz(-2.1482936962284063) q[13];
ry(2.0942576233991774) q[14];
rz(2.0988636363557975) q[14];
ry(-2.6642113486050767) q[15];
rz(-1.3827886021684144) q[15];
ry(-1.9196095024888438) q[16];
rz(2.3011775364930083) q[16];
ry(1.5306991084990313) q[17];
rz(0.5295331393748506) q[17];
ry(-2.6339155298993324) q[18];
rz(2.601617747010634) q[18];
ry(0.25684744769315054) q[19];
rz(0.09299089187358843) q[19];
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
ry(-3.1175283222467196) q[0];
rz(0.8892691387159797) q[0];
ry(0.05785018398934216) q[1];
rz(0.1819827771383162) q[1];
ry(3.0189083722641215) q[2];
rz(-0.4141805884258097) q[2];
ry(-0.11179063284344883) q[3];
rz(2.569943137204723) q[3];
ry(2.0901405610860504) q[4];
rz(2.918184068454904) q[4];
ry(-1.905587493262758) q[5];
rz(-0.5133956037410705) q[5];
ry(-1.8128925031697474) q[6];
rz(-2.345267575665971) q[6];
ry(-0.4178343026571698) q[7];
rz(-1.8331197487097874) q[7];
ry(3.1306381599294535) q[8];
rz(-0.7233457365513541) q[8];
ry(-3.082694943007944) q[9];
rz(2.966355629582703) q[9];
ry(-0.00044086495765851023) q[10];
rz(-0.10163327359749519) q[10];
ry(-2.075789091837643) q[11];
rz(-0.03532243939574229) q[11];
ry(1.7089270290017842) q[12];
rz(1.64544806572387) q[12];
ry(3.116785302547998) q[13];
rz(-1.712632445920974) q[13];
ry(2.675279938639603) q[14];
rz(-0.3033096557576603) q[14];
ry(3.0160542343531005) q[15];
rz(1.413969181536607) q[15];
ry(-2.118486768626947) q[16];
rz(-2.6382419625941567) q[16];
ry(-2.880432265105677) q[17];
rz(-1.7413293717637783) q[17];
ry(-1.7834416889863116) q[18];
rz(-2.3139650877749243) q[18];
ry(-0.4574006783649711) q[19];
rz(1.289254626543309) q[19];
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
ry(-1.1352468548908696) q[0];
rz(0.8630892735186061) q[0];
ry(-0.5596574599015208) q[1];
rz(1.6545150790341667) q[1];
ry(-0.09647176494395486) q[2];
rz(-1.0219530863910782) q[2];
ry(-0.1323865523113934) q[3];
rz(-0.8145197900605581) q[3];
ry(-0.803456578547026) q[4];
rz(-2.9991574241828483) q[4];
ry(1.5247128828518024) q[5];
rz(3.130600969595656) q[5];
ry(-3.117264068507717) q[6];
rz(-0.1197424139524692) q[6];
ry(2.153218960748606) q[7];
rz(-2.7778829214624055) q[7];
ry(1.3026737698247777) q[8];
rz(-2.385178277614295) q[8];
ry(-1.3542082806683715) q[9];
rz(-2.9016977030246376) q[9];
ry(-0.8530655527742819) q[10];
rz(1.4305683187413114) q[10];
ry(-2.7659688188140654) q[11];
rz(-3.1206186432144243) q[11];
ry(3.062261605848006) q[12];
rz(1.279046230575985) q[12];
ry(-1.5112830373854376) q[13];
rz(-0.07793030490446412) q[13];
ry(1.7395038196867427) q[14];
rz(2.8637811119588283) q[14];
ry(0.1767281849183906) q[15];
rz(2.1087907382621127) q[15];
ry(1.707068407207613) q[16];
rz(-2.6696993083912615) q[16];
ry(0.373765302752874) q[17];
rz(-2.772380274270236) q[17];
ry(-0.9497346091639773) q[18];
rz(-0.8859982983878035) q[18];
ry(2.8433951987865917) q[19];
rz(1.5994586978444427) q[19];
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
ry(-1.2670992078831314) q[0];
rz(-2.6600291317066493) q[0];
ry(3.1188316342338496) q[1];
rz(2.532328109659621) q[1];
ry(0.4802054937052941) q[2];
rz(1.6161640156365653) q[2];
ry(2.0413941318879987) q[3];
rz(-1.0517048126099429) q[3];
ry(-1.6222809225442674) q[4];
rz(-0.09276756056260883) q[4];
ry(1.8449103560510096) q[5];
rz(1.5457029831046354) q[5];
ry(0.1380095688259119) q[6];
rz(1.0001504671224435) q[6];
ry(0.2015647852980189) q[7];
rz(0.6567993594420258) q[7];
ry(0.11811365225431737) q[8];
rz(2.5096472257299745) q[8];
ry(-0.054948614878144626) q[9];
rz(-2.216178849975078) q[9];
ry(3.0940036778715942) q[10];
rz(-1.2255596016499144) q[10];
ry(-1.2277610165368895) q[11];
rz(-0.057121590916725744) q[11];
ry(-0.02313071276922981) q[12];
rz(-2.7340384852821495) q[12];
ry(3.0749120461898327) q[13];
rz(0.3304291778347075) q[13];
ry(1.7607161746991784) q[14];
rz(1.4547983093759713) q[14];
ry(-3.108457225490851) q[15];
rz(0.7623075294765984) q[15];
ry(0.1332438381926924) q[16];
rz(2.292232720505369) q[16];
ry(1.9980397707228388) q[17];
rz(0.3165796741640487) q[17];
ry(0.5531063152006644) q[18];
rz(1.1006267747362324) q[18];
ry(3.0183325063077824) q[19];
rz(0.5695482584973658) q[19];
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
ry(2.86158692557196) q[0];
rz(0.6478768371967205) q[0];
ry(-2.8586579261885703) q[1];
rz(-2.487820776479163) q[1];
ry(0.1901406099658844) q[2];
rz(-2.6110020649583427) q[2];
ry(-1.5045738415081038) q[3];
rz(-2.9333128769335355) q[3];
ry(2.0783882457596405) q[4];
rz(-1.6249072234876734) q[4];
ry(1.6051996794798873) q[5];
rz(0.269544146209416) q[5];
ry(0.7554513421452462) q[6];
rz(-2.065074305358881) q[6];
ry(-2.4596494591690057) q[7];
rz(0.7923530025581204) q[7];
ry(1.6341128355175796) q[8];
rz(-0.8716464769761595) q[8];
ry(0.7714962948423976) q[9];
rz(-2.9720745234530175) q[9];
ry(0.8348252490505615) q[10];
rz(2.037670650814061) q[10];
ry(1.2716650299172283) q[11];
rz(3.0748608404126325) q[11];
ry(1.706687946526259) q[12];
rz(0.09056237737088932) q[12];
ry(-0.04899107735320966) q[13];
rz(-0.5003586339156437) q[13];
ry(-0.039352599689757455) q[14];
rz(0.6954788652116691) q[14];
ry(-0.06286268861865785) q[15];
rz(1.0126274544077225) q[15];
ry(2.2562913222932695) q[16];
rz(0.5933657770030044) q[16];
ry(2.881961667301067) q[17];
rz(0.47822551656216794) q[17];
ry(-2.5500056188776328) q[18];
rz(-1.9329804215513313) q[18];
ry(0.49859225145581787) q[19];
rz(2.1518490244481243) q[19];
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
ry(-2.389943260488483) q[0];
rz(-1.9856418126070334) q[0];
ry(-2.6235456525822527) q[1];
rz(-2.3267287198982634) q[1];
ry(-3.002789271329531) q[2];
rz(2.32331176711467) q[2];
ry(1.0751391467799598) q[3];
rz(-1.893919041001141) q[3];
ry(0.0938803376695984) q[4];
rz(-1.6704756846569353) q[4];
ry(-0.04538993435778601) q[5];
rz(1.818377751516679) q[5];
ry(-2.9115786989930488) q[6];
rz(3.10738959702946) q[6];
ry(0.7173775395945399) q[7];
rz(2.933888796418536) q[7];
ry(0.10215290095522711) q[8];
rz(-2.1741319240324852) q[8];
ry(-2.7702818674103855) q[9];
rz(0.37821561493756306) q[9];
ry(-2.5203138189882086) q[10];
rz(-3.09957195404161) q[10];
ry(-2.639466363402993) q[11];
rz(-0.08493430461250018) q[11];
ry(-1.9890509796983398) q[12];
rz(0.9492614097262511) q[12];
ry(0.2866713230941677) q[13];
rz(-0.09945673289461253) q[13];
ry(0.24733093592253308) q[14];
rz(-2.14719859683544) q[14];
ry(-2.994951350992554) q[15];
rz(-0.02631245188665154) q[15];
ry(0.08508694171897702) q[16];
rz(2.352875596985738) q[16];
ry(-2.777599129122623) q[17];
rz(-0.9094207400536698) q[17];
ry(2.5580985615426832) q[18];
rz(0.29756486603988236) q[18];
ry(1.4909756285567846) q[19];
rz(-1.0148297280633463) q[19];
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
ry(2.9125433073361524) q[0];
rz(-2.5402007598348857) q[0];
ry(2.021245050589565) q[1];
rz(-0.5920516413130321) q[1];
ry(1.454361984315896) q[2];
rz(-2.071499230661965) q[2];
ry(2.5981971075141814) q[3];
rz(0.05446042835762582) q[3];
ry(-0.8903217946045788) q[4];
rz(0.9762918713191062) q[4];
ry(3.1352082928398586) q[5];
rz(-2.64633334396552) q[5];
ry(1.0907508152110486) q[6];
rz(2.284191446730647) q[6];
ry(-2.415824832542926) q[7];
rz(-0.5853849908265758) q[7];
ry(1.1785745437869004) q[8];
rz(-0.7643609194210772) q[8];
ry(-1.3207219313208065) q[9];
rz(0.6819873722605249) q[9];
ry(1.05754965425026) q[10];
rz(-0.2764961449053553) q[10];
ry(-1.3057811942589277) q[11];
rz(2.952268421464961) q[11];
ry(0.007841066915100642) q[12];
rz(-1.2452319993069052) q[12];
ry(0.18125900219635135) q[13];
rz(-0.4920779040673464) q[13];
ry(-0.9325147450539353) q[14];
rz(-0.9087780122764659) q[14];
ry(2.951634211886558) q[15];
rz(-0.24099036141263586) q[15];
ry(2.416752913631109) q[16];
rz(-1.251454137701531) q[16];
ry(2.687580619251098) q[17];
rz(-2.253903369056828) q[17];
ry(2.979357707066807) q[18];
rz(0.7434810878537412) q[18];
ry(-0.628455966834025) q[19];
rz(-0.05533282046993013) q[19];
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
ry(-1.8749435514055621) q[0];
rz(1.991275857873692) q[0];
ry(1.2734158117942278) q[1];
rz(2.742136045498494) q[1];
ry(2.211617921524018) q[2];
rz(0.31776246663365476) q[2];
ry(2.9166922395614505) q[3];
rz(-0.215299386914355) q[3];
ry(3.0682933682576166) q[4];
rz(0.6222987083116233) q[4];
ry(3.1133048368692293) q[5];
rz(2.726510706783935) q[5];
ry(-0.24361187310486887) q[6];
rz(1.2679299807414004) q[6];
ry(3.0990473437240613) q[7];
rz(-0.8101259818427484) q[7];
ry(-0.38362975866321486) q[8];
rz(-0.22324761264218565) q[8];
ry(0.5374210491403913) q[9];
rz(3.022910654451423) q[9];
ry(-2.1261322500925264) q[10];
rz(-0.08384938262712006) q[10];
ry(-3.024970667309272) q[11];
rz(-2.9119237545514753) q[11];
ry(-0.7950582541761027) q[12];
rz(-1.1252369495456809) q[12];
ry(-0.058507579488910624) q[13];
rz(2.080868255374349) q[13];
ry(0.0013648233693608347) q[14];
rz(-0.706500281194071) q[14];
ry(0.3233089729453375) q[15];
rz(0.15182412476163396) q[15];
ry(-0.07472907860683886) q[16];
rz(1.0054910248756752) q[16];
ry(-1.3228114713745418) q[17];
rz(0.15405773665542188) q[17];
ry(-0.012503345610558547) q[18];
rz(-0.09501213134924015) q[18];
ry(-1.2833057501922678) q[19];
rz(2.877159465446537) q[19];
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
ry(1.3605365248967374) q[0];
rz(1.5265789196846873) q[0];
ry(-2.9592656041736958) q[1];
rz(2.6150046418227073) q[1];
ry(2.534979197516956) q[2];
rz(0.7909607081608538) q[2];
ry(2.080749740482421) q[3];
rz(0.40425085760160145) q[3];
ry(-2.9493044841768463) q[4];
rz(-1.8945997838556) q[4];
ry(-2.8067950574739657) q[5];
rz(2.7130911720373163) q[5];
ry(-1.7231920006300523) q[6];
rz(-2.187801324984474) q[6];
ry(0.6779345371036173) q[7];
rz(0.2965856627229759) q[7];
ry(2.603247708313095) q[8];
rz(-2.5799847852720577) q[8];
ry(-2.881738707866656) q[9];
rz(-3.0910144877510777) q[9];
ry(-0.3391471810077757) q[10];
rz(3.123898851366853) q[10];
ry(0.19263996193317282) q[11];
rz(0.056526156793534454) q[11];
ry(2.4424726049219267) q[12];
rz(0.10302291326001216) q[12];
ry(-0.23056177708355904) q[13];
rz(0.02632060562111763) q[13];
ry(-1.5632154621034182) q[14];
rz(1.935206888778822) q[14];
ry(-1.358825668882587) q[15];
rz(1.5580187495629263) q[15];
ry(2.447649176474625) q[16];
rz(3.0963869938227977) q[16];
ry(-0.8605196942615025) q[17];
rz(-2.4125690503486514) q[17];
ry(0.20060301993830887) q[18];
rz(1.5217130221168345) q[18];
ry(-1.1134658539839686) q[19];
rz(-0.9424632231903434) q[19];
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
ry(1.0599857825522425) q[0];
rz(2.192675696538707) q[0];
ry(2.705809363513385) q[1];
rz(-0.5335005060321648) q[1];
ry(0.6208373240970507) q[2];
rz(0.8963242755611278) q[2];
ry(-2.851648143907468) q[3];
rz(2.2669297978958065) q[3];
ry(-0.08134700370132565) q[4];
rz(-0.21493049513494888) q[4];
ry(-0.03039279979564924) q[5];
rz(-2.189390226992481) q[5];
ry(0.29964604763791464) q[6];
rz(-0.21740397608007192) q[6];
ry(-2.7576917003322396) q[7];
rz(0.8270637817517329) q[7];
ry(-2.5402680755633766) q[8];
rz(-0.7296344093607675) q[8];
ry(0.6945673690425903) q[9];
rz(1.7014960221758082) q[9];
ry(1.6551678804218275) q[10];
rz(-0.6944656578256366) q[10];
ry(-0.07862863335733443) q[11];
rz(-2.3235818130637464) q[11];
ry(0.21331410874652834) q[12];
rz(-0.5760984045031252) q[12];
ry(-3.0134807908957577) q[13];
rz(0.048244364758604164) q[13];
ry(0.007429173034856795) q[14];
rz(-0.052287951518968256) q[14];
ry(-2.9727756621862826) q[15];
rz(-1.6827482293402372) q[15];
ry(-1.5898549976022442) q[16];
rz(-1.583031863607402) q[16];
ry(0.9530548185788186) q[17];
rz(-0.36639297915103874) q[17];
ry(2.4493874778131017) q[18];
rz(1.650529152838822) q[18];
ry(-1.6924001978623018) q[19];
rz(-2.5216400516434647) q[19];
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
ry(1.6644877549449841) q[0];
rz(-1.652735670559542) q[0];
ry(1.7877616699373065) q[1];
rz(-2.033115269574884) q[1];
ry(1.6965449031868545) q[2];
rz(2.2479543707647034) q[2];
ry(1.3141419684992013) q[3];
rz(-0.23661955464426046) q[3];
ry(-0.9369152788914006) q[4];
rz(-1.7872297934747683) q[4];
ry(2.8341090402361884) q[5];
rz(-0.7209771006158819) q[5];
ry(-2.057284373120379) q[6];
rz(-0.8817761562192518) q[6];
ry(-0.25802815155671954) q[7];
rz(-2.5174630766557846) q[7];
ry(2.660014589926097) q[8];
rz(-0.2592351789419664) q[8];
ry(-2.929543803352611) q[9];
rz(-2.2865137406529668) q[9];
ry(0.8176859223977848) q[10];
rz(2.393317928889318) q[10];
ry(1.3312945712672128) q[11];
rz(-0.7789198337630244) q[11];
ry(-1.9989707815336573) q[12];
rz(0.586009004008191) q[12];
ry(0.4078026110295987) q[13];
rz(-0.26408143993307664) q[13];
ry(2.816706706222243) q[14];
rz(-2.860309807801125) q[14];
ry(0.7561495378571869) q[15];
rz(2.3096089382835006) q[15];
ry(1.5777428405751313) q[16];
rz(2.101291400814069) q[16];
ry(1.5603598424424687) q[17];
rz(1.5675766669501696) q[17];
ry(-1.0699335193089863) q[18];
rz(0.1333379790411565) q[18];
ry(2.868746244459174) q[19];
rz(0.4882530758167034) q[19];
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
ry(-1.885895824224298) q[0];
rz(1.3867159109498486) q[0];
ry(-2.975963737553072) q[1];
rz(-0.7450867799589704) q[1];
ry(-0.0356583261724861) q[2];
rz(0.9722259025336478) q[2];
ry(2.9889409235365405) q[3];
rz(-2.762632991956545) q[3];
ry(-0.03745232219158545) q[4];
rz(-0.8614548370631877) q[4];
ry(3.1110811084031664) q[5];
rz(1.2269721132531437) q[5];
ry(0.24190233228169122) q[6];
rz(1.568493502558811) q[6];
ry(2.9670111817846987) q[7];
rz(2.6900924999550804) q[7];
ry(0.41622980360232553) q[8];
rz(-3.0318257615997415) q[8];
ry(0.03142762043314473) q[9];
rz(-1.10710832515314) q[9];
ry(0.1548246153174518) q[10];
rz(2.923998984622902) q[10];
ry(0.10576145288559556) q[11];
rz(-1.1043806581216176) q[11];
ry(-3.022552814426606) q[12];
rz(-0.5159477943817737) q[12];
ry(3.0182992129561232) q[13];
rz(-0.012598319930742006) q[13];
ry(0.018982648214227012) q[14];
rz(2.9480058400437685) q[14];
ry(-0.007129780082106407) q[15];
rz(2.504129312422418) q[15];
ry(0.0013186770118744917) q[16];
rz(1.2550880214411944) q[16];
ry(0.18269981345780212) q[17];
rz(0.007677702440998857) q[17];
ry(1.5733167089435725) q[18];
rz(-1.5724032338200375) q[18];
ry(-0.7478070734764737) q[19];
rz(1.2691665420732399) q[19];
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
ry(1.3944688257792337) q[0];
rz(2.9041291880252165) q[0];
ry(-1.7082341534663916) q[1];
rz(0.5235355224655196) q[1];
ry(0.7289732385012639) q[2];
rz(-0.9764367209456791) q[2];
ry(2.496424101990732) q[3];
rz(-0.39319184480741515) q[3];
ry(-2.316803098252715) q[4];
rz(2.610974517593207) q[4];
ry(3.1201724599336504) q[5];
rz(2.6755803250234647) q[5];
ry(0.802430737453237) q[6];
rz(-1.5982819889557156) q[6];
ry(1.7375523431226727) q[7];
rz(0.9492763030131048) q[7];
ry(-0.9665290657116102) q[8];
rz(1.01785383154498) q[8];
ry(-1.460157874396506) q[9];
rz(1.6155576152531) q[9];
ry(1.0142453943718026) q[10];
rz(2.0499769489911444) q[10];
ry(-2.686186857490986) q[11];
rz(1.1729923215320204) q[11];
ry(-2.4281856223821734) q[12];
rz(2.5122725143103457) q[12];
ry(-1.360806980490996) q[13];
rz(0.06697270298106517) q[13];
ry(-0.3269489877501588) q[14];
rz(-0.5671546876114579) q[14];
ry(-1.6578597475657277) q[15];
rz(1.411084271312122) q[15];
ry(3.1341329637356834) q[16];
rz(-0.7471107170261497) q[16];
ry(-1.565170718021089) q[17];
rz(1.2609534897632457) q[17];
ry(1.573222691624737) q[18];
rz(1.3628767061578682) q[18];
ry(-3.140603695704734) q[19];
rz(-1.115857684054519) q[19];