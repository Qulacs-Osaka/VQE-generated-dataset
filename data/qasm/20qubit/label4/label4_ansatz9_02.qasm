OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-0.5738435996754179) q[0];
ry(2.457759370323144) q[1];
cx q[0],q[1];
ry(-2.613816369279125) q[0];
ry(3.087446983058324) q[1];
cx q[0],q[1];
ry(-0.48216688228293497) q[2];
ry(-0.8631629754396278) q[3];
cx q[2],q[3];
ry(-0.24544048419555808) q[2];
ry(2.435718994451972) q[3];
cx q[2],q[3];
ry(2.8726843405251814) q[4];
ry(-1.4146215938039688) q[5];
cx q[4],q[5];
ry(-1.9215368884842003) q[4];
ry(1.7705560784139724) q[5];
cx q[4],q[5];
ry(-0.7619540758467309) q[6];
ry(-0.7676686174772014) q[7];
cx q[6],q[7];
ry(1.2553081814364617) q[6];
ry(-0.7736902752099695) q[7];
cx q[6],q[7];
ry(0.8930726559045369) q[8];
ry(2.951631264842312) q[9];
cx q[8],q[9];
ry(1.4272903117875226) q[8];
ry(-1.347685075592591) q[9];
cx q[8],q[9];
ry(-3.007034408648828) q[10];
ry(-0.9043164948345757) q[11];
cx q[10],q[11];
ry(-2.3377639205263114) q[10];
ry(-2.420999372326787) q[11];
cx q[10],q[11];
ry(0.45936014567646044) q[12];
ry(0.7970380703086093) q[13];
cx q[12],q[13];
ry(1.8694306215516008) q[12];
ry(-2.3615023167109337) q[13];
cx q[12],q[13];
ry(2.3287249227948177) q[14];
ry(-0.7603829162038842) q[15];
cx q[14],q[15];
ry(1.3287014398244477) q[14];
ry(1.0073711182452199) q[15];
cx q[14],q[15];
ry(1.6081668443147663) q[16];
ry(2.222128208357086) q[17];
cx q[16],q[17];
ry(-0.6849574049436434) q[16];
ry(-2.5531328286942436) q[17];
cx q[16],q[17];
ry(2.008096422168755) q[18];
ry(1.195667997572491) q[19];
cx q[18],q[19];
ry(1.0183273513042783) q[18];
ry(-1.6387705572732298) q[19];
cx q[18],q[19];
ry(0.5698743902339851) q[0];
ry(-0.11170392738282597) q[2];
cx q[0],q[2];
ry(-0.35378576380247034) q[0];
ry(1.8408247993228972) q[2];
cx q[0],q[2];
ry(2.4249541852546392) q[2];
ry(-1.8094296261476073) q[4];
cx q[2],q[4];
ry(-2.0649569292695533) q[2];
ry(1.612146616542712) q[4];
cx q[2],q[4];
ry(-0.03597615703574271) q[4];
ry(-0.9603132350926781) q[6];
cx q[4],q[6];
ry(3.1403521064930455) q[4];
ry(-0.0005103179228598264) q[6];
cx q[4],q[6];
ry(0.5513473331225919) q[6];
ry(-0.7041345365662375) q[8];
cx q[6],q[8];
ry(-0.18867336436494447) q[6];
ry(2.1707838474312804) q[8];
cx q[6],q[8];
ry(-3.068273646584858) q[8];
ry(-2.226851292161069) q[10];
cx q[8],q[10];
ry(2.0271192293645854) q[8];
ry(3.072126788068173) q[10];
cx q[8],q[10];
ry(-2.9078466057240835) q[10];
ry(-0.0990198914710625) q[12];
cx q[10],q[12];
ry(-0.19637144856765065) q[10];
ry(2.9646951586935852) q[12];
cx q[10],q[12];
ry(0.7122092326787017) q[12];
ry(2.117742350422528) q[14];
cx q[12],q[14];
ry(-1.8113258489296689) q[12];
ry(-1.3462524350717917) q[14];
cx q[12],q[14];
ry(1.849971339524699) q[14];
ry(-2.90030057004385) q[16];
cx q[14],q[16];
ry(3.078419791189668) q[14];
ry(-3.0836687984938966) q[16];
cx q[14],q[16];
ry(-0.46569466452031516) q[16];
ry(-2.416910664556895) q[18];
cx q[16],q[18];
ry(1.5197572057213318) q[16];
ry(2.6460568316659625) q[18];
cx q[16],q[18];
ry(2.8552351512713106) q[1];
ry(0.11898700571355367) q[3];
cx q[1],q[3];
ry(1.9630532281969204) q[1];
ry(2.171834197753282) q[3];
cx q[1],q[3];
ry(0.8813598294295096) q[3];
ry(-0.5800179097960583) q[5];
cx q[3],q[5];
ry(-1.8573543083124022) q[3];
ry(-1.0587174038812828) q[5];
cx q[3],q[5];
ry(0.8493214949526786) q[5];
ry(1.2528481098462205) q[7];
cx q[5],q[7];
ry(-2.3054778284955915) q[5];
ry(2.592175333126185) q[7];
cx q[5],q[7];
ry(-0.9612346267906938) q[7];
ry(0.807942507607415) q[9];
cx q[7],q[9];
ry(3.094082131901415) q[7];
ry(0.002380756228884353) q[9];
cx q[7],q[9];
ry(2.5304911410142816) q[9];
ry(-2.7887417386318436) q[11];
cx q[9],q[11];
ry(2.966471580251642) q[9];
ry(-0.1401054884030852) q[11];
cx q[9],q[11];
ry(0.7557438145653723) q[11];
ry(-0.8735377191607263) q[13];
cx q[11],q[13];
ry(1.8262277294237137) q[11];
ry(-1.4189387946500043) q[13];
cx q[11],q[13];
ry(-1.215113780583142) q[13];
ry(1.1112174696628858) q[15];
cx q[13],q[15];
ry(-2.6066755007594935) q[13];
ry(2.5766367578757317) q[15];
cx q[13],q[15];
ry(1.3297067601943366) q[15];
ry(1.402850179698289) q[17];
cx q[15],q[17];
ry(-1.9132408121577553) q[15];
ry(2.4624484611844455) q[17];
cx q[15],q[17];
ry(0.33599254932313544) q[17];
ry(1.6364404804791377) q[19];
cx q[17],q[19];
ry(-2.440908860406796) q[17];
ry(-0.04858015162734172) q[19];
cx q[17],q[19];
ry(-2.1442271974649545) q[0];
ry(2.0325301788489583) q[3];
cx q[0],q[3];
ry(2.938751280770955) q[0];
ry(-1.9992648228907202) q[3];
cx q[0],q[3];
ry(-0.4779206597257249) q[1];
ry(-2.6854708517420454) q[2];
cx q[1],q[2];
ry(-0.29875029819523924) q[1];
ry(-1.8834343374932407) q[2];
cx q[1],q[2];
ry(2.8148834156930476) q[2];
ry(-2.313472021820458) q[5];
cx q[2],q[5];
ry(-3.0570487411803136) q[2];
ry(-1.2049692731227286) q[5];
cx q[2],q[5];
ry(0.659697342485754) q[3];
ry(0.8045722638578533) q[4];
cx q[3],q[4];
ry(-0.8417012311941905) q[3];
ry(2.0041455544498357) q[4];
cx q[3],q[4];
ry(-2.87949399636401) q[4];
ry(-2.4942341497082214) q[7];
cx q[4],q[7];
ry(-0.08941511433263738) q[4];
ry(-1.7032492667202923) q[7];
cx q[4],q[7];
ry(-2.2574281616558682) q[5];
ry(-1.4257082642700887) q[6];
cx q[5],q[6];
ry(2.973393193874882) q[5];
ry(-0.004723139174922508) q[6];
cx q[5],q[6];
ry(-1.9272777565117698) q[6];
ry(1.7312592717666404) q[9];
cx q[6],q[9];
ry(-2.141119492090011) q[6];
ry(-0.44724675446760465) q[9];
cx q[6],q[9];
ry(1.6617845029245517) q[7];
ry(1.4059870389578588) q[8];
cx q[7],q[8];
ry(0.37721760466060683) q[7];
ry(2.79443884570021) q[8];
cx q[7],q[8];
ry(-2.4531019350528247) q[8];
ry(1.4964280537726316) q[11];
cx q[8],q[11];
ry(-0.9931961738360668) q[8];
ry(-1.9810468607879697) q[11];
cx q[8],q[11];
ry(-0.17643789378016453) q[9];
ry(-2.250198169993558) q[10];
cx q[9],q[10];
ry(-0.006577694895533171) q[9];
ry(3.1261428503092272) q[10];
cx q[9],q[10];
ry(-0.29190849676069675) q[10];
ry(-1.4622313280357062) q[13];
cx q[10],q[13];
ry(2.7045686774547657) q[10];
ry(1.1343520070621222) q[13];
cx q[10],q[13];
ry(-1.1692019333704593) q[11];
ry(-1.6713093747290253) q[12];
cx q[11],q[12];
ry(-1.3181887057553352) q[11];
ry(1.1289148729192329) q[12];
cx q[11],q[12];
ry(1.942735710991168) q[12];
ry(-1.3112263084085773) q[15];
cx q[12],q[15];
ry(-0.5265288292381098) q[12];
ry(-0.3049104231677209) q[15];
cx q[12],q[15];
ry(1.3874058541993752) q[13];
ry(-2.183232195851173) q[14];
cx q[13],q[14];
ry(0.02998069460418673) q[13];
ry(-3.0930352158076) q[14];
cx q[13],q[14];
ry(2.4709672772109106) q[14];
ry(2.848474249166714) q[17];
cx q[14],q[17];
ry(-1.5216537012193214) q[14];
ry(2.3781822956101215) q[17];
cx q[14],q[17];
ry(0.23722493373028786) q[15];
ry(0.25603651092358337) q[16];
cx q[15],q[16];
ry(-1.1825553135647797) q[15];
ry(3.1316342181089984) q[16];
cx q[15],q[16];
ry(-1.824290200132876) q[16];
ry(-1.2938825160400906) q[19];
cx q[16],q[19];
ry(-0.8156170959073394) q[16];
ry(1.7478379419709713) q[19];
cx q[16],q[19];
ry(-2.8764880392770285) q[17];
ry(0.15187401616441573) q[18];
cx q[17],q[18];
ry(1.7281432568085835) q[17];
ry(3.1226829900470237) q[18];
cx q[17],q[18];
ry(1.3954956575598472) q[0];
ry(0.9795713193444453) q[1];
cx q[0],q[1];
ry(-0.6292750091521775) q[0];
ry(-2.5539136162075264) q[1];
cx q[0],q[1];
ry(-0.31396015489157575) q[2];
ry(-0.5952869888963411) q[3];
cx q[2],q[3];
ry(1.5799804994266378) q[2];
ry(1.7320389174841642) q[3];
cx q[2],q[3];
ry(-1.2280994197139954) q[4];
ry(1.4684504927726811) q[5];
cx q[4],q[5];
ry(3.0885425554622676) q[4];
ry(0.7004126232471409) q[5];
cx q[4],q[5];
ry(-1.8930947324627168) q[6];
ry(2.384713371251103) q[7];
cx q[6],q[7];
ry(0.6208073462424348) q[6];
ry(2.315409204154841) q[7];
cx q[6],q[7];
ry(1.6450518389868334) q[8];
ry(-1.0144282910378488) q[9];
cx q[8],q[9];
ry(-1.2206629897963754) q[8];
ry(-2.4742722579461343) q[9];
cx q[8],q[9];
ry(-1.5520404826726262) q[10];
ry(-2.4563730998861786) q[11];
cx q[10],q[11];
ry(-1.8145073663336744) q[10];
ry(-2.720648270480034) q[11];
cx q[10],q[11];
ry(-0.39889437668182115) q[12];
ry(1.1630903054931279) q[13];
cx q[12],q[13];
ry(-1.4940215162585215) q[12];
ry(-2.2324495106921027) q[13];
cx q[12],q[13];
ry(-0.03227859136685218) q[14];
ry(0.7638515799262455) q[15];
cx q[14],q[15];
ry(0.5034428846657025) q[14];
ry(1.6628626984570067) q[15];
cx q[14],q[15];
ry(-2.3005913826298054) q[16];
ry(-0.021159407911100582) q[17];
cx q[16],q[17];
ry(-0.03057675281719252) q[16];
ry(2.3944532130030876) q[17];
cx q[16],q[17];
ry(2.7522072034697382) q[18];
ry(-1.9557105122879916) q[19];
cx q[18],q[19];
ry(-1.5837999801353844) q[18];
ry(-1.5895542362113773) q[19];
cx q[18],q[19];
ry(-1.8988571365445024) q[0];
ry(2.5105104995509415) q[2];
cx q[0],q[2];
ry(3.026075507782192) q[0];
ry(1.4967084487974993) q[2];
cx q[0],q[2];
ry(-2.538449963374107) q[2];
ry(-2.7838906994625336) q[4];
cx q[2],q[4];
ry(-3.09338080453969) q[2];
ry(-1.9511187941159598) q[4];
cx q[2],q[4];
ry(0.4612356089403198) q[4];
ry(2.1813576511735135) q[6];
cx q[4],q[6];
ry(-0.0015903180984562226) q[4];
ry(3.1408362483114796) q[6];
cx q[4],q[6];
ry(1.448702150075202) q[6];
ry(-2.2147375965965668) q[8];
cx q[6],q[8];
ry(1.8147947297910367) q[6];
ry(-2.719900006371311) q[8];
cx q[6],q[8];
ry(0.751863949532134) q[8];
ry(-1.115402728793284) q[10];
cx q[8],q[10];
ry(-3.1399741550853593) q[8];
ry(3.14076894582438) q[10];
cx q[8],q[10];
ry(-2.8499401280867014) q[10];
ry(-0.5469195106823195) q[12];
cx q[10],q[12];
ry(-2.713651496759016) q[10];
ry(-1.2591567431959232) q[12];
cx q[10],q[12];
ry(2.2579597074333804) q[12];
ry(2.968096496029777) q[14];
cx q[12],q[14];
ry(3.121813608590441) q[12];
ry(-0.011342244780854216) q[14];
cx q[12],q[14];
ry(-1.9519489721521084) q[14];
ry(0.545827415352309) q[16];
cx q[14],q[16];
ry(3.000461695167948) q[14];
ry(-3.128412099320496) q[16];
cx q[14],q[16];
ry(2.797505817534453) q[16];
ry(-0.37519082639209017) q[18];
cx q[16],q[18];
ry(-1.3377604440909332) q[16];
ry(-0.7940126515849799) q[18];
cx q[16],q[18];
ry(-0.8565040397041637) q[1];
ry(-2.574661432805997) q[3];
cx q[1],q[3];
ry(2.860074365885299) q[1];
ry(-2.6458235855077956) q[3];
cx q[1],q[3];
ry(-0.23695934077005276) q[3];
ry(-0.23101374374185735) q[5];
cx q[3],q[5];
ry(-0.5368916915532864) q[3];
ry(0.014690126908272205) q[5];
cx q[3],q[5];
ry(2.6442912668603844) q[5];
ry(1.4160328281432084) q[7];
cx q[5],q[7];
ry(0.007578528767846038) q[5];
ry(-3.1372897216901356) q[7];
cx q[5],q[7];
ry(-2.5470043093391763) q[7];
ry(-2.4898372825584945) q[9];
cx q[7],q[9];
ry(2.8491148788165903) q[7];
ry(3.026243199817754) q[9];
cx q[7],q[9];
ry(1.5241980387728589) q[9];
ry(-1.4564748632545923) q[11];
cx q[9],q[11];
ry(-0.009381780315555588) q[9];
ry(-3.1414351665019087) q[11];
cx q[9],q[11];
ry(2.1506462980422167) q[11];
ry(-2.361364923069421) q[13];
cx q[11],q[13];
ry(0.9672381286031966) q[11];
ry(-2.4563372166314315) q[13];
cx q[11],q[13];
ry(-2.313086053209217) q[13];
ry(1.4802569763985192) q[15];
cx q[13],q[15];
ry(0.008617822450732504) q[13];
ry(-0.01436634779124812) q[15];
cx q[13],q[15];
ry(-1.5209201670310373) q[15];
ry(0.8926269503491424) q[17];
cx q[15],q[17];
ry(-1.2036136400409214) q[15];
ry(1.4741146520889974) q[17];
cx q[15],q[17];
ry(1.8296778176334803) q[17];
ry(-1.1547291272195639) q[19];
cx q[17],q[19];
ry(-2.317572088907554) q[17];
ry(3.0182109994716297) q[19];
cx q[17],q[19];
ry(-1.8250544786856007) q[0];
ry(-2.690733438871837) q[3];
cx q[0],q[3];
ry(-0.30929035075077405) q[0];
ry(0.6673372851706162) q[3];
cx q[0],q[3];
ry(2.5242682715737823) q[1];
ry(1.906573407181479) q[2];
cx q[1],q[2];
ry(-3.0289973016991505) q[1];
ry(-2.826992833831097) q[2];
cx q[1],q[2];
ry(1.4790818255873288) q[2];
ry(1.094622862938711) q[5];
cx q[2],q[5];
ry(-0.2683593594813711) q[2];
ry(2.354061784406691) q[5];
cx q[2],q[5];
ry(2.566844734845363) q[3];
ry(-1.2222697589458926) q[4];
cx q[3],q[4];
ry(-3.020641578032499) q[3];
ry(-1.7201725508826897) q[4];
cx q[3],q[4];
ry(1.4884502096108418) q[4];
ry(-2.9786026388131073) q[7];
cx q[4],q[7];
ry(-0.0023929354705272132) q[4];
ry(-3.1384426263453102) q[7];
cx q[4],q[7];
ry(2.312907039584857) q[5];
ry(0.8380316779601298) q[6];
cx q[5],q[6];
ry(-3.127866575746987) q[5];
ry(-3.1346105725228006) q[6];
cx q[5],q[6];
ry(0.623221485622666) q[6];
ry(1.021356369283034) q[9];
cx q[6],q[9];
ry(-2.098058619685397) q[6];
ry(-0.35196090869096164) q[9];
cx q[6],q[9];
ry(1.6835378210423517) q[7];
ry(-2.3835271808497476) q[8];
cx q[7],q[8];
ry(-0.6694566937338111) q[7];
ry(2.804530437534764) q[8];
cx q[7],q[8];
ry(0.4547602237304691) q[8];
ry(2.5096083650408585) q[11];
cx q[8],q[11];
ry(-3.1091266823578843) q[8];
ry(-3.126135317154593) q[11];
cx q[8],q[11];
ry(-0.6681729297228531) q[9];
ry(-0.44664666731303276) q[10];
cx q[9],q[10];
ry(-3.134372272199857) q[9];
ry(-3.140368677650749) q[10];
cx q[9],q[10];
ry(-1.3197870097013782) q[10];
ry(-1.4674671319005987) q[13];
cx q[10],q[13];
ry(-1.8165608350402715) q[10];
ry(-0.03228044861337942) q[13];
cx q[10],q[13];
ry(-0.2928539149662033) q[11];
ry(-1.7355252621388768) q[12];
cx q[11],q[12];
ry(-2.5975678181057518) q[11];
ry(-2.787063491919862) q[12];
cx q[11],q[12];
ry(1.4368057458350378) q[12];
ry(-0.748055986533072) q[15];
cx q[12],q[15];
ry(-3.1391210284439417) q[12];
ry(0.010163908683987975) q[15];
cx q[12],q[15];
ry(0.6234345779976191) q[13];
ry(-1.9209506840270565) q[14];
cx q[13],q[14];
ry(-3.1406984672024274) q[13];
ry(-3.1244166915409326) q[14];
cx q[13],q[14];
ry(2.2871999590267733) q[14];
ry(2.4297617106061917) q[17];
cx q[14],q[17];
ry(-0.7546536446865071) q[14];
ry(0.2755178376109387) q[17];
cx q[14],q[17];
ry(-2.275724528221719) q[15];
ry(2.0970654219086473) q[16];
cx q[15],q[16];
ry(0.9114752541108636) q[15];
ry(0.0017922975452968615) q[16];
cx q[15],q[16];
ry(3.077265675281069) q[16];
ry(-2.8947879034703017) q[19];
cx q[16],q[19];
ry(-1.124032671088493) q[16];
ry(-0.19389692266815722) q[19];
cx q[16],q[19];
ry(2.295136200306569) q[17];
ry(1.189410830549522) q[18];
cx q[17],q[18];
ry(2.795159473813972) q[17];
ry(-2.6493573166918156) q[18];
cx q[17],q[18];
ry(0.8511997498160434) q[0];
ry(1.3563948090977878) q[1];
cx q[0],q[1];
ry(0.36637215057217104) q[0];
ry(-2.7652518886142134) q[1];
cx q[0],q[1];
ry(1.371750154097155) q[2];
ry(0.7476712730686632) q[3];
cx q[2],q[3];
ry(-1.0499775242324532) q[2];
ry(-0.3149522522160302) q[3];
cx q[2],q[3];
ry(-0.5371794863531045) q[4];
ry(-0.15284305300864265) q[5];
cx q[4],q[5];
ry(1.3876892280809803) q[4];
ry(2.0032154111271225) q[5];
cx q[4],q[5];
ry(-2.0882529549640703) q[6];
ry(0.5418685415034473) q[7];
cx q[6],q[7];
ry(1.766299957980996) q[6];
ry(-1.1684589595508292) q[7];
cx q[6],q[7];
ry(2.489591096253134) q[8];
ry(-1.5910983557457854) q[9];
cx q[8],q[9];
ry(1.794736006940469) q[8];
ry(-1.9739671147058173) q[9];
cx q[8],q[9];
ry(-0.9475427894524691) q[10];
ry(-2.9694966797151543) q[11];
cx q[10],q[11];
ry(1.3421158287295158) q[10];
ry(-1.7596861441669762) q[11];
cx q[10],q[11];
ry(2.807013125081484) q[12];
ry(2.28032355103685) q[13];
cx q[12],q[13];
ry(1.7440964511196082) q[12];
ry(1.5342631753405855) q[13];
cx q[12],q[13];
ry(2.6119352515790366) q[14];
ry(1.5607834258952904) q[15];
cx q[14],q[15];
ry(-3.0742373026302676) q[14];
ry(-1.3096817838936072) q[15];
cx q[14],q[15];
ry(2.8576545162607907) q[16];
ry(0.2304577298780065) q[17];
cx q[16],q[17];
ry(0.7038182821791261) q[16];
ry(2.818633912667758) q[17];
cx q[16],q[17];
ry(-2.4140283888932745) q[18];
ry(-1.855529253260559) q[19];
cx q[18],q[19];
ry(2.4566772696823636) q[18];
ry(2.3624159292116356) q[19];
cx q[18],q[19];
ry(1.937801914757633) q[0];
ry(1.8166249260866865) q[2];
cx q[0],q[2];
ry(-3.086201516234578) q[0];
ry(-0.7592918681294618) q[2];
cx q[0],q[2];
ry(0.9486514985232017) q[2];
ry(-2.2690158457221035) q[4];
cx q[2],q[4];
ry(0.3118022821588484) q[2];
ry(1.4365791493343953) q[4];
cx q[2],q[4];
ry(1.6581170456541872) q[4];
ry(0.9067533342766945) q[6];
cx q[4],q[6];
ry(0.0029307625206014265) q[4];
ry(0.0023639475625020167) q[6];
cx q[4],q[6];
ry(-0.11021289814054125) q[6];
ry(2.0127384272117235) q[8];
cx q[6],q[8];
ry(-0.7803372203774996) q[6];
ry(-2.895053361225474) q[8];
cx q[6],q[8];
ry(0.10348895158038561) q[8];
ry(-0.1186155347393902) q[10];
cx q[8],q[10];
ry(-3.1301309799010815) q[8];
ry(0.00949132026185447) q[10];
cx q[8],q[10];
ry(0.35722008321253185) q[10];
ry(1.9826398431158143) q[12];
cx q[10],q[12];
ry(-0.086475114104851) q[10];
ry(2.860809878781205) q[12];
cx q[10],q[12];
ry(1.970323886844831) q[12];
ry(-3.003173612518105) q[14];
cx q[12],q[14];
ry(0.006307783599740269) q[12];
ry(-3.128739745307789) q[14];
cx q[12],q[14];
ry(0.6914785973511863) q[14];
ry(-1.5196729382744671) q[16];
cx q[14],q[16];
ry(3.1329895987269514) q[14];
ry(-3.1331263312912125) q[16];
cx q[14],q[16];
ry(-3.0570760093495752) q[16];
ry(1.3171831602197468) q[18];
cx q[16],q[18];
ry(0.15988178721096613) q[16];
ry(0.07323415451350712) q[18];
cx q[16],q[18];
ry(-0.8331502532380116) q[1];
ry(-2.110932629930975) q[3];
cx q[1],q[3];
ry(2.6669013132905826) q[1];
ry(3.069370237280227) q[3];
cx q[1],q[3];
ry(0.748782524124019) q[3];
ry(2.82274897376691) q[5];
cx q[3],q[5];
ry(0.9004601269842062) q[3];
ry(-0.8545390065368429) q[5];
cx q[3],q[5];
ry(-1.5908229057035146) q[5];
ry(-1.0368176487164198) q[7];
cx q[5],q[7];
ry(1.63387296807146) q[5];
ry(0.001991738207908555) q[7];
cx q[5],q[7];
ry(1.549284779123585) q[7];
ry(-0.16539763759102385) q[9];
cx q[7],q[9];
ry(2.050572421339697) q[7];
ry(-2.5167633387533517) q[9];
cx q[7],q[9];
ry(1.6057802908102863) q[9];
ry(-3.095666927876728) q[11];
cx q[9],q[11];
ry(-0.008243575171656303) q[9];
ry(-3.1388266750864107) q[11];
cx q[9],q[11];
ry(2.9935933892113833) q[11];
ry(-2.2953785743456816) q[13];
cx q[11],q[13];
ry(1.5395611710657136) q[11];
ry(1.5411153220252864) q[13];
cx q[11],q[13];
ry(2.6770867982396505) q[13];
ry(-3.1348524512804663) q[15];
cx q[13],q[15];
ry(3.1408399304656918) q[13];
ry(3.1298730247984055) q[15];
cx q[13],q[15];
ry(-2.1783416842247516) q[15];
ry(-1.45437267379506) q[17];
cx q[15],q[17];
ry(-1.7573334004130563) q[15];
ry(0.19456818330521752) q[17];
cx q[15],q[17];
ry(1.8438119065476357) q[17];
ry(0.06032300278736265) q[19];
cx q[17],q[19];
ry(2.9910449969045194) q[17];
ry(-2.948674235004143) q[19];
cx q[17],q[19];
ry(2.848743951235284) q[0];
ry(0.7623756520593467) q[3];
cx q[0],q[3];
ry(0.10114128266353983) q[0];
ry(-2.3443089584563754) q[3];
cx q[0],q[3];
ry(-1.9261475885914603) q[1];
ry(0.2237916147871415) q[2];
cx q[1],q[2];
ry(1.1305529307379043) q[1];
ry(-2.0670075840629782) q[2];
cx q[1],q[2];
ry(1.8770473165857995) q[2];
ry(-2.6089720639041407) q[5];
cx q[2],q[5];
ry(-3.1400104664976936) q[2];
ry(-2.6396895749802383) q[5];
cx q[2],q[5];
ry(0.09678439671004035) q[3];
ry(-2.356974873271709) q[4];
cx q[3],q[4];
ry(-1.4282856096199295) q[3];
ry(-1.5150335578997014) q[4];
cx q[3],q[4];
ry(1.4953897187705447) q[4];
ry(-0.10773571545602412) q[7];
cx q[4],q[7];
ry(3.139818088542047) q[4];
ry(3.1366377101822143) q[7];
cx q[4],q[7];
ry(3.081800638837843) q[5];
ry(2.6439025991708647) q[6];
cx q[5],q[6];
ry(-1.570418938353665) q[5];
ry(3.1388894908031144) q[6];
cx q[5],q[6];
ry(0.12742311502781423) q[6];
ry(-1.57905659303949) q[9];
cx q[6],q[9];
ry(-0.8201924471099493) q[6];
ry(-3.1415474955389278) q[9];
cx q[6],q[9];
ry(-0.7267293821829309) q[7];
ry(-3.0441876642998236) q[8];
cx q[7],q[8];
ry(-0.5366538758208437) q[7];
ry(-2.9420300155530534) q[8];
cx q[7],q[8];
ry(1.2094266434474055) q[8];
ry(-0.5497724046925274) q[11];
cx q[8],q[11];
ry(3.141246015196588) q[8];
ry(-0.0016519026004875317) q[11];
cx q[8],q[11];
ry(1.1471606209697407) q[9];
ry(-2.5214567524629845) q[10];
cx q[9],q[10];
ry(-0.0010994761540153624) q[9];
ry(3.1386689764541367) q[10];
cx q[9],q[10];
ry(0.31437447603697155) q[10];
ry(2.6599063782827517) q[13];
cx q[10],q[13];
ry(-1.2270762439404308) q[10];
ry(-0.10674112284220615) q[13];
cx q[10],q[13];
ry(-0.8163988683095917) q[11];
ry(2.851208263893738) q[12];
cx q[11],q[12];
ry(0.5258587865329778) q[11];
ry(-0.6026909571336869) q[12];
cx q[11],q[12];
ry(-0.6355789993818861) q[12];
ry(2.892449111823857) q[15];
cx q[12],q[15];
ry(-3.1384142657749092) q[12];
ry(0.005810137755802725) q[15];
cx q[12],q[15];
ry(1.0972701748698857) q[13];
ry(-1.006814235474518) q[14];
cx q[13],q[14];
ry(0.0013346875724344898) q[13];
ry(0.465868798510904) q[14];
cx q[13],q[14];
ry(-0.7072692315609793) q[14];
ry(-1.5188742192004914) q[17];
cx q[14],q[17];
ry(-0.15861799717699082) q[14];
ry(-0.011945160974991924) q[17];
cx q[14],q[17];
ry(-0.3894335398363751) q[15];
ry(-0.4610486553456301) q[16];
cx q[15],q[16];
ry(1.6778997011384047) q[15];
ry(3.0693935502692336) q[16];
cx q[15],q[16];
ry(3.0753196773066014) q[16];
ry(2.8604170404550104) q[19];
cx q[16],q[19];
ry(-1.9473433122927022) q[16];
ry(1.1324190831631729) q[19];
cx q[16],q[19];
ry(1.5501910524942435) q[17];
ry(-1.1175048668968266) q[18];
cx q[17],q[18];
ry(2.2459686554638223) q[17];
ry(0.9931394008714376) q[18];
cx q[17],q[18];
ry(-1.0869987547169915) q[0];
ry(0.3934073251561001) q[1];
cx q[0],q[1];
ry(-2.851093915162333) q[0];
ry(-0.5624449181080794) q[1];
cx q[0],q[1];
ry(-2.321525526774719) q[2];
ry(0.6449830313543349) q[3];
cx q[2],q[3];
ry(-1.9413663318555228) q[2];
ry(-1.7144430412213385) q[3];
cx q[2],q[3];
ry(-1.3170368659545648) q[4];
ry(-0.9038570335203264) q[5];
cx q[4],q[5];
ry(0.015471860900570178) q[4];
ry(1.2825160523176309) q[5];
cx q[4],q[5];
ry(1.3563612605111048) q[6];
ry(0.8552452536929397) q[7];
cx q[6],q[7];
ry(2.8764326100741315) q[6];
ry(-0.1818286577906445) q[7];
cx q[6],q[7];
ry(-0.02155543756571977) q[8];
ry(-0.8050520911758247) q[9];
cx q[8],q[9];
ry(0.642272119304191) q[8];
ry(-1.493002365542927) q[9];
cx q[8],q[9];
ry(-1.3947213179043993) q[10];
ry(-1.9369979561507957) q[11];
cx q[10],q[11];
ry(-0.26088760452107) q[10];
ry(-2.4087884260952706) q[11];
cx q[10],q[11];
ry(1.8747542098348742) q[12];
ry(-1.5133432485836782) q[13];
cx q[12],q[13];
ry(-2.4068842450841395) q[12];
ry(0.035971943610011776) q[13];
cx q[12],q[13];
ry(-2.155332656794509) q[14];
ry(0.5713982821208345) q[15];
cx q[14],q[15];
ry(0.13491699786018801) q[14];
ry(1.6267392876594566) q[15];
cx q[14],q[15];
ry(-1.4862619397711725) q[16];
ry(-1.5834451693760085) q[17];
cx q[16],q[17];
ry(-1.21183691855939) q[16];
ry(-3.006054752131954) q[17];
cx q[16],q[17];
ry(0.7932539595575515) q[18];
ry(-1.745229046106777) q[19];
cx q[18],q[19];
ry(-0.6380393099643475) q[18];
ry(-1.3987597189055005) q[19];
cx q[18],q[19];
ry(1.2446096290738877) q[0];
ry(2.902414150036031) q[2];
cx q[0],q[2];
ry(2.359892778520945) q[0];
ry(-1.3883473257035943) q[2];
cx q[0],q[2];
ry(2.866145236345509) q[2];
ry(1.4173602470268243) q[4];
cx q[2],q[4];
ry(2.455880240559154) q[2];
ry(-3.073306199441237) q[4];
cx q[2],q[4];
ry(-1.7714343564115522) q[4];
ry(-0.6474936714532894) q[6];
cx q[4],q[6];
ry(3.1413280317034293) q[4];
ry(0.003198158275854013) q[6];
cx q[4],q[6];
ry(0.8028771432075787) q[6];
ry(2.037600974288774) q[8];
cx q[6],q[8];
ry(-1.8770232446402024) q[6];
ry(0.4358237132267734) q[8];
cx q[6],q[8];
ry(-2.0681953681111125) q[8];
ry(1.0949656843014033) q[10];
cx q[8],q[10];
ry(3.1181349194064953) q[8];
ry(3.1378417034974406) q[10];
cx q[8],q[10];
ry(-1.8436070894209173) q[10];
ry(-1.9989547623835304) q[12];
cx q[10],q[12];
ry(-3.0179086510869046) q[10];
ry(0.9253282752096316) q[12];
cx q[10],q[12];
ry(-0.6572906973824386) q[12];
ry(-1.3987263123822564) q[14];
cx q[12],q[14];
ry(3.134047358737201) q[12];
ry(-0.021283269631298438) q[14];
cx q[12],q[14];
ry(-2.6781347632511117) q[14];
ry(-0.84028176137759) q[16];
cx q[14],q[16];
ry(-3.1163226496388625) q[14];
ry(3.1151865572371453) q[16];
cx q[14],q[16];
ry(2.9047430315176848) q[16];
ry(-3.073499194674719) q[18];
cx q[16],q[18];
ry(0.0740391885241187) q[16];
ry(2.7570605175115284) q[18];
cx q[16],q[18];
ry(2.4329140989146234) q[1];
ry(1.3009683921992523) q[3];
cx q[1],q[3];
ry(1.4514662589265332) q[1];
ry(0.7022773592520348) q[3];
cx q[1],q[3];
ry(0.6849000045401975) q[3];
ry(1.1805580793248647) q[5];
cx q[3],q[5];
ry(-3.0907168618565044) q[3];
ry(-0.16289549711749207) q[5];
cx q[3],q[5];
ry(2.1738921992627427) q[5];
ry(0.5351288703640137) q[7];
cx q[5],q[7];
ry(0.24470589088215408) q[5];
ry(-0.009383564241833398) q[7];
cx q[5],q[7];
ry(0.20349240774619204) q[7];
ry(1.3774193546220221) q[9];
cx q[7],q[9];
ry(0.8117597159273365) q[7];
ry(0.007573492052649173) q[9];
cx q[7],q[9];
ry(-1.0418835254360426) q[9];
ry(-3.0158862800618875) q[11];
cx q[9],q[11];
ry(0.009603660497194797) q[9];
ry(-3.1365504170022755) q[11];
cx q[9],q[11];
ry(-0.2697130063899227) q[11];
ry(-0.17992620489630298) q[13];
cx q[11],q[13];
ry(2.653478524810369) q[11];
ry(-2.0585353985746693) q[13];
cx q[11],q[13];
ry(-2.9853948624705557) q[13];
ry(2.7551541813393143) q[15];
cx q[13],q[15];
ry(0.010193875459774827) q[13];
ry(-0.04217041836812019) q[15];
cx q[13],q[15];
ry(1.0400267718196226) q[15];
ry(-3.00541143929771) q[17];
cx q[15],q[17];
ry(-1.7453087679809047) q[15];
ry(0.009212365100107256) q[17];
cx q[15],q[17];
ry(-0.5470793186148484) q[17];
ry(-0.6801996067076175) q[19];
cx q[17],q[19];
ry(1.830234314234975) q[17];
ry(1.7627175984277432) q[19];
cx q[17],q[19];
ry(2.3542950940924254) q[0];
ry(1.3865967983737713) q[3];
cx q[0],q[3];
ry(1.1336939833560988) q[0];
ry(1.437642825813139) q[3];
cx q[0],q[3];
ry(-2.991961123235388) q[1];
ry(-1.6071606541390677) q[2];
cx q[1],q[2];
ry(-2.568636118969925) q[1];
ry(-0.3292871988044357) q[2];
cx q[1],q[2];
ry(0.5709341278247236) q[2];
ry(0.6268359973019287) q[5];
cx q[2],q[5];
ry(0.0028647372122049303) q[2];
ry(1.7930382081343927) q[5];
cx q[2],q[5];
ry(0.49816249439215365) q[3];
ry(0.8447992086534809) q[4];
cx q[3],q[4];
ry(-0.3891807583979716) q[3];
ry(1.4445004043473726) q[4];
cx q[3],q[4];
ry(-1.5594473941897895) q[4];
ry(-2.4367279814328042) q[7];
cx q[4],q[7];
ry(0.0035161084914220224) q[4];
ry(3.140823774123761) q[7];
cx q[4],q[7];
ry(-0.09162605470912599) q[5];
ry(-1.2433838522446568) q[6];
cx q[5],q[6];
ry(0.2546341876429237) q[5];
ry(-0.02607947183037229) q[6];
cx q[5],q[6];
ry(-2.221217828750909) q[6];
ry(-0.25522299554696964) q[9];
cx q[6],q[9];
ry(-0.44722200760873393) q[6];
ry(2.947652242906289) q[9];
cx q[6],q[9];
ry(-2.0276461475523404) q[7];
ry(-0.28295177888186385) q[8];
cx q[7],q[8];
ry(-0.9647980014519486) q[7];
ry(0.9435095867937523) q[8];
cx q[7],q[8];
ry(-2.36290060500057) q[8];
ry(-1.2344564368826363) q[11];
cx q[8],q[11];
ry(-0.0072913710164908) q[8];
ry(-3.1183633880722166) q[11];
cx q[8],q[11];
ry(1.033514687068613) q[9];
ry(0.8816968892024111) q[10];
cx q[9],q[10];
ry(0.36180518548872787) q[9];
ry(-1.9999399738834702) q[10];
cx q[9],q[10];
ry(-1.6287322695253568) q[10];
ry(1.6717296255906449) q[13];
cx q[10],q[13];
ry(-3.1314946939321975) q[10];
ry(3.1013101793572404) q[13];
cx q[10],q[13];
ry(1.633265577003546) q[11];
ry(-1.2592006548869277) q[12];
cx q[11],q[12];
ry(-1.9648146130353386) q[11];
ry(2.1066303267536903) q[12];
cx q[11],q[12];
ry(0.992015336898695) q[12];
ry(-0.028450134746203837) q[15];
cx q[12],q[15];
ry(-0.02003485743049196) q[12];
ry(-0.018328236229044875) q[15];
cx q[12],q[15];
ry(1.9258198314681516) q[13];
ry(0.30734983578564434) q[14];
cx q[13],q[14];
ry(-2.7228277031896435) q[13];
ry(-0.3972783160250114) q[14];
cx q[13],q[14];
ry(-3.0037668542129836) q[14];
ry(-1.4782239803153066) q[17];
cx q[14],q[17];
ry(3.1249788131821488) q[14];
ry(-3.0078227849580013) q[17];
cx q[14],q[17];
ry(3.0902433641157283) q[15];
ry(0.3358117899452584) q[16];
cx q[15],q[16];
ry(0.9667677611936135) q[15];
ry(-0.03676824560104709) q[16];
cx q[15],q[16];
ry(-1.9742133060847622) q[16];
ry(-2.898774889654811) q[19];
cx q[16],q[19];
ry(-2.9489929941037105) q[16];
ry(-2.379586751174133) q[19];
cx q[16],q[19];
ry(-2.662662130673807) q[17];
ry(2.3913738136825424) q[18];
cx q[17],q[18];
ry(-2.3701863014419957) q[17];
ry(-0.6616960609078726) q[18];
cx q[17],q[18];
ry(0.634568060718295) q[0];
ry(-2.7134698778665247) q[1];
cx q[0],q[1];
ry(2.1802569944253323) q[0];
ry(2.3406358622113896) q[1];
cx q[0],q[1];
ry(-0.24965262514495296) q[2];
ry(-2.524620926118451) q[3];
cx q[2],q[3];
ry(-2.402332937294485) q[2];
ry(-2.667175138137924) q[3];
cx q[2],q[3];
ry(-0.5603425051298893) q[4];
ry(1.8600424298455194) q[5];
cx q[4],q[5];
ry(0.42220003685029805) q[4];
ry(0.5387465060954794) q[5];
cx q[4],q[5];
ry(0.8956825880862315) q[6];
ry(-1.4303991154509266) q[7];
cx q[6],q[7];
ry(-1.7394914514416788) q[6];
ry(-2.8091096076704) q[7];
cx q[6],q[7];
ry(-0.636711916904641) q[8];
ry(1.5693276432263943) q[9];
cx q[8],q[9];
ry(-1.6084239080220093) q[8];
ry(-0.31703077931759616) q[9];
cx q[8],q[9];
ry(3.055968907431606) q[10];
ry(-2.9484044175593813) q[11];
cx q[10],q[11];
ry(1.343603596554579) q[10];
ry(-1.559376695233028) q[11];
cx q[10],q[11];
ry(0.5580837971823693) q[12];
ry(1.6561949882531373) q[13];
cx q[12],q[13];
ry(1.630512024985122) q[12];
ry(1.5569518290451874) q[13];
cx q[12],q[13];
ry(-1.4806333232570474) q[14];
ry(1.4007752539336051) q[15];
cx q[14],q[15];
ry(-1.5227620198498129) q[14];
ry(1.6603650226238642) q[15];
cx q[14],q[15];
ry(-0.15472456462942452) q[16];
ry(-1.1931430286974054) q[17];
cx q[16],q[17];
ry(-2.980641803642781) q[16];
ry(1.628515392507981) q[17];
cx q[16],q[17];
ry(-2.453809003452416) q[18];
ry(-0.1190250030719735) q[19];
cx q[18],q[19];
ry(-0.277380962444151) q[18];
ry(-1.2976140115028825) q[19];
cx q[18],q[19];
ry(0.8752441605423575) q[0];
ry(-2.844893782716601) q[2];
cx q[0],q[2];
ry(-1.0281846184441052) q[0];
ry(2.303639427012692) q[2];
cx q[0],q[2];
ry(1.541121035829083) q[2];
ry(2.849245620300619) q[4];
cx q[2],q[4];
ry(-1.6517961327673996) q[2];
ry(-2.757675005157574) q[4];
cx q[2],q[4];
ry(-0.7266198813301159) q[4];
ry(-0.7245896136354446) q[6];
cx q[4],q[6];
ry(-0.00945156432340521) q[4];
ry(-3.140311936738502) q[6];
cx q[4],q[6];
ry(0.334997592417281) q[6];
ry(-2.472361767360627) q[8];
cx q[6],q[8];
ry(-3.103212452823802) q[6];
ry(-0.032313071203792454) q[8];
cx q[6],q[8];
ry(-1.6568033525605061) q[8];
ry(1.9061165303053045) q[10];
cx q[8],q[10];
ry(0.03706239718426934) q[8];
ry(-0.06706288821874404) q[10];
cx q[8],q[10];
ry(-0.15765486532052256) q[10];
ry(2.7707258891724638) q[12];
cx q[10],q[12];
ry(-3.130122949754164) q[10];
ry(3.1372091634027948) q[12];
cx q[10],q[12];
ry(1.625506461628756) q[12];
ry(-1.2119225162441944) q[14];
cx q[12],q[14];
ry(-0.06309923225253389) q[12];
ry(-2.9295055787600495) q[14];
cx q[12],q[14];
ry(1.4648301278866436) q[14];
ry(1.6584199138716889) q[16];
cx q[14],q[16];
ry(-0.12959401878877722) q[14];
ry(-3.093438258792453) q[16];
cx q[14],q[16];
ry(-1.516104526020862) q[16];
ry(-2.4332639474172884) q[18];
cx q[16],q[18];
ry(0.08223914298170573) q[16];
ry(-1.3775213720466342) q[18];
cx q[16],q[18];
ry(-1.3576467805041226) q[1];
ry(-0.4033357734368436) q[3];
cx q[1],q[3];
ry(-1.1194124022497338) q[1];
ry(3.0760361898749773) q[3];
cx q[1],q[3];
ry(1.7804126909555786) q[3];
ry(0.48934694006534846) q[5];
cx q[3],q[5];
ry(-0.000997014515115926) q[3];
ry(3.1391901197189624) q[5];
cx q[3],q[5];
ry(-0.47349631818652504) q[5];
ry(2.461552608317711) q[7];
cx q[5],q[7];
ry(-3.1135915259054148) q[5];
ry(0.02253148566650917) q[7];
cx q[5],q[7];
ry(1.0101364312755088) q[7];
ry(-2.004296972746684) q[9];
cx q[7],q[9];
ry(0.039459029454029376) q[7];
ry(-3.1234476935023396) q[9];
cx q[7],q[9];
ry(-2.9431861597781186) q[9];
ry(-3.0080129215965243) q[11];
cx q[9],q[11];
ry(3.1266598456998014) q[9];
ry(3.0019553943817843) q[11];
cx q[9],q[11];
ry(-2.2021484601073436) q[11];
ry(-2.4237821723864914) q[13];
cx q[11],q[13];
ry(0.041766675873338606) q[11];
ry(3.1309267294845085) q[13];
cx q[11],q[13];
ry(-2.2430251014341236) q[13];
ry(-1.9112872930754028) q[15];
cx q[13],q[15];
ry(-3.138335184138922) q[13];
ry(-3.0746885299208753) q[15];
cx q[13],q[15];
ry(-2.716862151615075) q[15];
ry(1.9589942956304247) q[17];
cx q[15],q[17];
ry(0.12658080566626848) q[15];
ry(0.16978647449841855) q[17];
cx q[15],q[17];
ry(-3.1236793688458726) q[17];
ry(-0.14251686780965045) q[19];
cx q[17],q[19];
ry(2.815471203390352) q[17];
ry(1.9800943790434895) q[19];
cx q[17],q[19];
ry(-0.04671633272276665) q[0];
ry(2.8545552941958228) q[3];
cx q[0],q[3];
ry(-0.17393522674515907) q[0];
ry(-0.5939093310828528) q[3];
cx q[0],q[3];
ry(-0.8379224961915508) q[1];
ry(-2.124990760326358) q[2];
cx q[1],q[2];
ry(-1.5547240379104599) q[1];
ry(0.7527508558119669) q[2];
cx q[1],q[2];
ry(0.7722429369396283) q[2];
ry(1.4511213732637955) q[5];
cx q[2],q[5];
ry(-0.0023740570403134598) q[2];
ry(-0.0010185594946511651) q[5];
cx q[2],q[5];
ry(2.213808348880262) q[3];
ry(1.0580074489750342) q[4];
cx q[3],q[4];
ry(0.04793350710529421) q[3];
ry(1.549739868588797) q[4];
cx q[3],q[4];
ry(-1.5877364965544025) q[4];
ry(1.6624826676767377) q[7];
cx q[4],q[7];
ry(0.018885002487282566) q[4];
ry(-3.1401698980501473) q[7];
cx q[4],q[7];
ry(2.513754351105202) q[5];
ry(-2.8715028826063835) q[6];
cx q[5],q[6];
ry(0.020580070368537488) q[5];
ry(3.1080414878976157) q[6];
cx q[5],q[6];
ry(2.478887780815687) q[6];
ry(2.4696110969684613) q[9];
cx q[6],q[9];
ry(-3.126887664840304) q[6];
ry(3.1299026780234853) q[9];
cx q[6],q[9];
ry(-2.0987018185655444) q[7];
ry(-2.393523376556696) q[8];
cx q[7],q[8];
ry(1.473139804097403) q[7];
ry(0.08743266798851224) q[8];
cx q[7],q[8];
ry(-1.9670009478688835) q[8];
ry(-2.3157670561985118) q[11];
cx q[8],q[11];
ry(3.121301492570338) q[8];
ry(0.04852801565245013) q[11];
cx q[8],q[11];
ry(2.6712681875284248) q[9];
ry(0.18981800766490628) q[10];
cx q[9],q[10];
ry(-0.6057402125111873) q[9];
ry(-1.4235550845363962) q[10];
cx q[9],q[10];
ry(-0.4627994857159603) q[10];
ry(0.5391818076511461) q[13];
cx q[10],q[13];
ry(-0.010244163043528076) q[10];
ry(0.009376739377646715) q[13];
cx q[10],q[13];
ry(1.5452740979499695) q[11];
ry(0.1912724436847988) q[12];
cx q[11],q[12];
ry(3.061738823375504) q[11];
ry(-2.7107853034349385) q[12];
cx q[11],q[12];
ry(-1.9028070100476002) q[12];
ry(-2.5698956499532444) q[15];
cx q[12],q[15];
ry(-0.05158501413127465) q[12];
ry(0.08514317474508193) q[15];
cx q[12],q[15];
ry(-0.5166451269204817) q[13];
ry(-1.3187546404347472) q[14];
cx q[13],q[14];
ry(-2.075659756475611) q[13];
ry(1.8023291344744508) q[14];
cx q[13],q[14];
ry(1.506833077037343) q[14];
ry(-1.9291345024514204) q[17];
cx q[14],q[17];
ry(0.04013670399873424) q[14];
ry(-0.005608482497403675) q[17];
cx q[14],q[17];
ry(-1.3571450864406431) q[15];
ry(-1.5248845472168302) q[16];
cx q[15],q[16];
ry(1.4666635049640668) q[15];
ry(-2.1940347007224896) q[16];
cx q[15],q[16];
ry(-0.15235891586916575) q[16];
ry(2.4338878940334756) q[19];
cx q[16],q[19];
ry(-3.0795253864158973) q[16];
ry(0.08014548897146608) q[19];
cx q[16],q[19];
ry(-0.3056873608769514) q[17];
ry(-0.6310075674706014) q[18];
cx q[17],q[18];
ry(1.3565792707983382) q[17];
ry(1.3490520286355614) q[18];
cx q[17],q[18];
ry(-2.2228884624319747) q[0];
ry(1.1826423874740744) q[1];
ry(2.392769250405466) q[2];
ry(0.006080370461722284) q[3];
ry(1.7270499835503128) q[4];
ry(0.8456146306805037) q[5];
ry(-1.0493767669264062) q[6];
ry(-1.0928412548785094) q[7];
ry(-1.1894844256742116) q[8];
ry(1.6210503387133812) q[9];
ry(-2.6748663416646434) q[10];
ry(-1.5851403246195614) q[11];
ry(1.6766056814324566) q[12];
ry(1.529244187060698) q[13];
ry(1.6369677777716425) q[14];
ry(1.5801797769119321) q[15];
ry(-2.9627727844250504) q[16];
ry(0.2305197250231189) q[17];
ry(-3.091002917348714) q[18];
ry(0.10226257725892493) q[19];