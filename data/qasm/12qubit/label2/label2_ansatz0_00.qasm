OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.7198387301123335) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.6227715052823729) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.03935878903660896) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-2.1815332135958982) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(1.5358219583329802) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.4321243288938873) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.005211567096107498) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.07473437237281796) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-1.0289440491054958) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.060102688800716564) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(0.03901123877578794) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(0.1574562789966881) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(1.5877535052641423) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(0.006567653509831078) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-1.8794055081927374) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(-1.7557432477131825) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(1.5644622342459422) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.13893688549186994) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.3599319166507587) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(3.599093070361786e-05) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.12844609827394834) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-0.0006647883524674166) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-0.000929533837570689) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(0.028657010314943967) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(-0.5005584761204829) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(0.7147533403218188) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(-0.9829685957427) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(1.7096638013381702) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(2.9387313042224634) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.42325362657810306) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(1.3075826030243165) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-1.5022805010533642e-05) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(1.0074553043841765) q[11];
cx q[10],q[11];
rx(-0.4954406147620811) q[0];
rz(-0.21286617668546307) q[0];
rx(-1.3923473305238623) q[1];
rz(-0.2545864983417898) q[1];
rx(-0.44032480640933275) q[2];
rz(0.2940957240275061) q[2];
rx(2.960044247501108) q[3];
rz(-1.7800425863191789) q[3];
rx(1.5681360446205201) q[4];
rz(0.07198275058044873) q[4];
rx(-0.03413572769326939) q[5];
rz(0.1623124253673023) q[5];
rx(0.5425759009548877) q[6];
rz(-1.1023155393476487) q[6];
rx(0.788282206955957) q[7];
rz(0.12162160123267846) q[7];
rx(0.002240261920451788) q[8];
rz(-1.2796620946502377) q[8];
rx(-0.003973556990800628) q[9];
rz(0.45365400091088937) q[9];
rx(0.7535121110072255) q[10];
rz(0.0022400519457428654) q[10];
rx(-0.6680033122728012) q[11];
rz(2.034867538653479) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(-1.6100228236624936) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.08775890114393228) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.11331309837377529) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.40658022006977024) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(1.1575617992852674) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.5066154693837978) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.0025032837375891927) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.005187701471165309) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.004007892228556903) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.03027674386253132) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.0002070435201479981) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(0.001563708520049865) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-0.1684503834932968) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(0.0049102951394856915) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(0.032559418467110975) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(-1.543806246123641) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(1.9830203399447162) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(-1.1923182156591186) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.728138312322225) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(1.8353659838006096) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.12092305238763378) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-0.0006322413551998245) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(5.7326539791558273e-05) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(-0.00021173280498732383) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(1.994149581922142) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(0.9383823105840543) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(1.12891430569373) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(1.7762377726970406) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-3.140198947779228) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(0.0019479728181780896) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(0.17643169040608253) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-0.14154168678938703) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.222931534091876) q[11];
cx q[10],q[11];
rx(0.8329138343603386) q[0];
rz(-1.0317328757772442) q[0];
rx(-2.398052544227088) q[1];
rz(-0.5316843882921792) q[1];
rx(0.33085481027005065) q[2];
rz(0.19405554199212305) q[2];
rx(0.0277491891733487) q[3];
rz(0.37449125019690543) q[3];
rx(2.1377977777732955) q[4];
rz(1.5636919459577825) q[4];
rx(-0.03242348294567698) q[5];
rz(0.05923330050292938) q[5];
rx(0.02958999754234127) q[6];
rz(-0.07053225181528734) q[6];
rx(0.27796003204015707) q[7];
rz(-1.5433003449386442) q[7];
rx(-5.25672383148291e-05) q[8];
rz(0.0953760528535573) q[8];
rx(0.0018487620974752073) q[9];
rz(-0.6950446821236527) q[9];
rx(-0.10217116623297667) q[10];
rz(1.5787954984952344) q[10];
rx(-1.0273331671484276) q[11];
rz(0.7381911929832465) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(-1.6953268314824526) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.015168050652209115) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-1.0091361608470018) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.21252592424568634) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(1.5403165517535984) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.055975653283486346) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.005361939950378428) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(1.5613303649034158) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.0034289572801792893) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(-0.048793563272310436) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.019782716094495) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-0.04878899065498663) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(0.01023828676238069) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-1.5805769424052787) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.01052969869196234) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(-0.08360932513596664) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(2.5861087790992046) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.061002737013406756) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.004146527732413094) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(1.6247560895290425) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.04557303566607939) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-0.047593996278253975) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(0.04791166846057676) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(-0.047693956454936036) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(0.7729180031095699) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(2.3679277885348737) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(0.7736327683409916) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(0.04739625490140266) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-3.0841832244846956) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(0.048307920603206836) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(-0.8064596119552676) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-0.7790419358703499) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.7163824720411293) q[11];
cx q[10],q[11];
rx(0.38815673584512966) q[0];
rz(-1.412127837419136) q[0];
rx(-0.780602092883813) q[1];
rz(-3.052032877483107) q[1];
rx(-0.15532865037860458) q[2];
rz(-0.9242291898472931) q[2];
rx(0.7910637462533031) q[3];
rz(0.5334225844584667) q[3];
rx(0.7899357893930765) q[4];
rz(0.5351333276888005) q[4];
rx(-2.2895671482643603) q[5];
rz(0.5280283291328507) q[5];
rx(-0.028838323607069578) q[6];
rz(0.7511341718775648) q[6];
rx(0.7241122459401155) q[7];
rz(0.4806586304763508) q[7];
rx(0.7251547054409874) q[8];
rz(0.4843709722559179) q[8];
rx(0.7221231475680491) q[9];
rz(0.48216016630468944) q[9];
rx(0.7441221601050523) q[10];
rz(0.4696473344675238) q[10];
rx(0.6862909863180352) q[11];
rz(0.4768604699199429) q[11];