OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.5707969155332124) q[0];
rz(0.5239656876261574) q[0];
ry(-1.5706475522686176) q[1];
rz(0.01120862398765201) q[1];
ry(-1.5727699957471728) q[2];
rz(-3.1320003539389494) q[2];
ry(-1.5708019992387607) q[3];
rz(-3.141546828255245) q[3];
ry(-1.5676218629073788) q[4];
rz(0.002734928294573277) q[4];
ry(1.5834468141187967) q[5];
rz(-2.609275673961671) q[5];
ry(-1.5476413898450085) q[6];
rz(1.5475979161430882) q[6];
ry(0.005426415292146381) q[7];
rz(-1.673691944703683) q[7];
ry(-3.0976885210797316) q[8];
rz(3.010195034565136) q[8];
ry(-1.570069632157253) q[9];
rz(-0.9112351122480342) q[9];
ry(0.004033220568857677) q[10];
rz(0.47063849138729374) q[10];
ry(-1.5714926873221016) q[11];
rz(0.848216213032817) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(3.1415863287435495) q[0];
rz(2.094757975996636) q[0];
ry(-0.013468734168402108) q[1];
rz(-1.5825530661334415) q[1];
ry(-2.9771598866144395) q[2];
rz(1.5513516461987615) q[2];
ry(1.53362275812942) q[3];
rz(1.5694745565955464) q[3];
ry(2.902598873916751) q[4];
rz(-1.57409644253638) q[4];
ry(0.0001428481260208026) q[5];
rz(-0.6856473797241317) q[5];
ry(-1.6438415532923152) q[6];
rz(2.5625980317739767) q[6];
ry(-1.6190800737570337) q[7];
rz(0.001961841213799522) q[7];
ry(3.094414770941071) q[8];
rz(0.7013101182254903) q[8];
ry(-5.898662914560759e-06) q[9];
rz(-0.5092141561556071) q[9];
ry(3.1407578469244473) q[10];
rz(-0.8252502053463445) q[10];
ry(3.1415907920761157) q[11];
rz(-0.09085498533405721) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.4909029135842193) q[0];
rz(-1.9009121925148942) q[0];
ry(0.09079054636897689) q[1];
rz(-1.5717119499420589) q[1];
ry(-0.0007030758244974676) q[2];
rz(1.4878203629656575) q[2];
ry(-0.013730370868683405) q[3];
rz(-1.5728431617875636) q[3];
ry(0.002371310316349951) q[4];
rz(-1.6066470253262883) q[4];
ry(-6.135532815623761e-06) q[5];
rz(-2.096017811678427) q[5];
ry(-0.005497630717577806) q[6];
rz(-2.9083284488826737) q[6];
ry(1.5738660588341569) q[7];
rz(1.2913158046884368) q[7];
ry(-8.043056351958455e-05) q[8];
rz(-2.1543091126249836) q[8];
ry(-0.0004949943612002627) q[9];
rz(-1.1358124219245682) q[9];
ry(-0.04740425491071164) q[10];
rz(3.1243969582862072) q[10];
ry(6.095534690331306e-07) q[11];
rz(-2.2072158386406695) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-2.174127592084483) q[0];
rz(-0.1920154513621577) q[0];
ry(-1.25963798033004) q[1];
rz(0.0006096287783616816) q[1];
ry(-3.123357899191859) q[2];
rz(1.4574178009474101) q[2];
ry(-0.1952186502465585) q[3];
rz(1.3714315854850474) q[3];
ry(2.2653598272221265) q[4];
rz(1.6069773338806446) q[4];
ry(0.31033409094041714) q[5];
rz(0.7049642317092832) q[5];
ry(-1.5956780830200312) q[6];
rz(-3.104946568804895) q[6];
ry(3.1097170824898717) q[7];
rz(1.0548963650239285) q[7];
ry(-1.528250889086469) q[8];
rz(0.018164071984682152) q[8];
ry(0.0013214892819189842) q[9];
rz(1.2517886693370892) q[9];
ry(-1.567980329199012) q[10];
rz(-0.14997485066887253) q[10];
ry(2.966579856801307) q[11];
rz(-1.575545607367709) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.5706447077551315) q[0];
rz(-1.7688814684405172) q[0];
ry(-1.5707803890972218) q[1];
rz(-0.5939242809063626) q[1];
ry(-2.280886309550625) q[2];
rz(1.5759978747870098) q[2];
ry(-3.133668321129143) q[3];
rz(-1.7725092676999972) q[3];
ry(0.018043395101568403) q[4];
rz(2.269372461844843) q[4];
ry(1.3049726168674216) q[5];
rz(-1.4572529265717609) q[5];
ry(-0.002675813779230474) q[6];
rz(-0.7324423225993718) q[6];
ry(3.1368342095414263) q[7];
rz(1.4048680273239484) q[7];
ry(0.028663240230855713) q[8];
rz(0.5992845970797638) q[8];
ry(-0.00014269888120553941) q[9];
rz(2.8625248185985104) q[9];
ry(0.010886165569161221) q[10];
rz(-1.4998151899484407) q[10];
ry(1.917992705860069) q[11];
rz(-0.8529974704320019) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.5711931033046462) q[0];
rz(0.0005649383190107214) q[0];
ry(1.322976812265065) q[1];
rz(0.034306282485400755) q[1];
ry(-3.092531241566995) q[2];
rz(3.1155399700535384) q[2];
ry(-1.62076803485333) q[3];
rz(-1.7359211269525694) q[3];
ry(-8.132841705066342e-05) q[4];
rz(-2.4982746970346073) q[4];
ry(0.0011905321181409079) q[5];
rz(1.28115634783991) q[5];
ry(4.3583178742565175e-05) q[6];
rz(-1.7692523699517297) q[6];
ry(-4.217413071602005e-05) q[7];
rz(-2.6282218657406045) q[7];
ry(-1.0366199538400773e-06) q[8];
rz(-0.6244683705566316) q[8];
ry(3.1415921694545585) q[9];
rz(3.128833993434091) q[9];
ry(3.1412377105394245) q[10];
rz(-0.8947472848852448) q[10];
ry(3.141587435595156) q[11];
rz(2.28866938056365) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.565969059770822) q[0];
rz(-1.7478954969029372) q[0];
ry(-3.13585993299003) q[1];
rz(0.49469708553512476) q[1];
ry(1.5843807685509343) q[2];
rz(0.28341696527340693) q[2];
ry(0.024561799832269936) q[3];
rz(-2.976193583481018) q[3];
ry(0.24224444236456844) q[4];
rz(1.3912233080472864) q[4];
ry(-0.956933410507424) q[5];
rz(0.4874109390992283) q[5];
ry(-3.016501710979113) q[6];
rz(-2.390544076713084) q[6];
ry(-0.1338599865882326) q[7];
rz(-2.5486425305996425) q[7];
ry(2.0323789105259866) q[8];
rz(-1.732563282687785) q[8];
ry(-2.3375763042492483) q[9];
rz(2.3318360211921894) q[9];
ry(-3.137756423962411) q[10];
rz(3.0937061066226157) q[10];
ry(-2.0111444670147165) q[11];
rz(-1.5706420008307327) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.5725579249611015) q[0];
rz(-1.5690165762697799) q[0];
ry(3.1391284945852025) q[1];
rz(0.5662343622398003) q[1];
ry(1.1915824399439163) q[2];
rz(-2.3904690353573295) q[2];
ry(-1.571286799745927) q[3];
rz(-1.77542823898464) q[3];
ry(-0.1433162457186315) q[4];
rz(-1.503038662461246) q[4];
ry(0.00011451329643108465) q[5];
rz(2.9538927735519147) q[5];
ry(3.1391895535983836) q[6];
rz(1.079059442150803) q[6];
ry(-3.1405140243959564) q[7];
rz(-0.21839468606280113) q[7];
ry(-0.00021300177349825589) q[8];
rz(-0.1346854907499928) q[8];
ry(-3.14158758751282) q[9];
rz(-0.8097408821730712) q[9];
ry(2.8510730931193877e-05) q[10];
rz(-2.400949177343103) q[10];
ry(1.365401737113817) q[11];
rz(3.0306476249752867) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.570240230756867) q[0];
rz(-1.57173131765102) q[0];
ry(-3.141572970012585) q[1];
rz(-1.870753432303136) q[1];
ry(3.14124291505919) q[2];
rz(0.9298313386394581) q[2];
ry(-0.00016164579719024627) q[3];
rz(-1.027356291769728) q[3];
ry(0.001735960250524471) q[4];
rz(-1.3024990391482438) q[4];
ry(-0.8088061572779425) q[5];
rz(1.570864746660968) q[5];
ry(-1.3192749901940915e-05) q[6];
rz(-1.3642979515436937) q[6];
ry(1.549229587105572e-05) q[7];
rz(-3.031997656294682) q[7];
ry(3.1415915524030456) q[8];
rz(2.948438008778526) q[8];
ry(0.762199351454945) q[9];
rz(-1.5708033444086318) q[9];
ry(0.0006043369805098316) q[10];
rz(1.5137118717654516) q[10];
ry(3.141585286814475) q[11];
rz(3.0306550084640063) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.5805885429212607) q[0];
rz(-1.4738462824494463) q[0];
ry(0.009311192324949147) q[1];
rz(1.7328862049444165) q[1];
ry(-1.5730953038257176) q[2];
rz(1.5734424700337648) q[2];
ry(1.58213870578186) q[3];
rz(3.062258110464487) q[3];
ry(-1.519082488220934) q[4];
rz(0.0009438314360741274) q[4];
ry(1.5732998621741157) q[5];
rz(1.5845426494290455) q[5];
ry(3.1415482017391723) q[6];
rz(-2.3820717170920225) q[6];
ry(3.14152582639088) q[7];
rz(-1.7423129315708932) q[7];
ry(0.00027320096103109403) q[8];
rz(-2.991727156586725) q[8];
ry(0.5940093097383199) q[9];
rz(1.5708001024273421) q[9];
ry(-0.0001392521131480251) q[10];
rz(1.6824812080154334) q[10];
ry(2.11220248093768) q[11];
rz(-1.5707879011385448) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.0005673968135372059) q[0];
rz(-1.6668867703416346) q[0];
ry(3.1399961710820254) q[1];
rz(2.5253170000218006) q[1];
ry(1.5742359834405695) q[2];
rz(0.007261885407439039) q[2];
ry(-1.568719788327102) q[3];
rz(1.5703241144782467) q[3];
ry(1.56541482453229) q[4];
rz(1.4705406935019913) q[4];
ry(-0.445746120060135) q[5];
rz(-0.01238857442596686) q[5];
ry(1.4846917921372924) q[6];
rz(1.7472295565825915) q[6];
ry(1.5181198916370988) q[7];
rz(-2.7951763336140867) q[7];
ry(-0.0023538885191656668) q[8];
rz(-0.3062171942737031) q[8];
ry(-1.034262520462171) q[9];
rz(1.5707912624895344) q[9];
ry(-0.43643744207373725) q[10];
rz(0.9798376909720408) q[10];
ry(-1.584509570822583) q[11];
rz(2.28862075036609) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.894017349018022) q[0];
rz(-2.089140460607951) q[0];
ry(2.2306974185636657) q[1];
rz(2.2264123255636683) q[1];
ry(2.4678884875627682) q[2];
rz(1.1381274453439212) q[2];
ry(-2.4677851737141387) q[3];
rz(-1.461252267965901) q[3];
ry(1.947372706474802) q[4];
rz(-1.3538980515657988) q[4];
ry(-1.57175856936869) q[5];
rz(1.529899111784328) q[5];
ry(0.00018325982765820102) q[6];
rz(1.618890094996817) q[6];
ry(3.141583317952353) q[7];
rz(0.18148354988889828) q[7];
ry(3.141387900644439) q[8];
rz(-1.481225436390563) q[8];
ry(-1.5730464805591229) q[9];
rz(1.7362442700671794) q[9];
ry(9.253108551376954e-06) q[10];
rz(-1.434705058880187) q[10];
ry(-3.1415860277670675) q[11];
rz(2.2143958803199784) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-3.1369816241714745) q[0];
rz(-0.5288414183616812) q[0];
ry(-1.5664450181207852) q[1];
rz(-1.5764416916214197) q[1];
ry(3.141314588959696) q[2];
rz(1.046550315081216) q[2];
ry(9.064270972466595e-05) q[3];
rz(1.1239287930654906) q[3];
ry(-3.140663005826438) q[4];
rz(1.6496556838747247) q[4];
ry(7.421928918738274e-05) q[5];
rz(1.6104387514783358) q[5];
ry(-3.141553909649253) q[6];
rz(0.10294471820227578) q[6];
ry(3.141549369344245) q[7];
rz(-0.05206831540151046) q[7];
ry(0.000108523259539511) q[8];
rz(1.430320656516339) q[8];
ry(-0.00011667404695803185) q[9];
rz(2.3000166388711722) q[9];
ry(0.00011542194374370042) q[10];
rz(-1.1085418580648039) q[10];
ry(-3.141474020767027) q[11];
rz(1.496584914801132) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(0.00942802165723065) q[0];
rz(2.7505468879392025) q[0];
ry(-1.5661017401149557) q[1];
rz(1.8072917773788968) q[1];
ry(-3.141567229613679) q[2];
rz(-1.655225035072407) q[2];
ry(0.00013695279716291052) q[3];
rz(-1.2366235475100256) q[3];
ry(3.1412569166719733) q[4];
rz(1.2653554749588627) q[4];
ry(-1.805998604803218) q[5];
rz(3.135533597098635) q[5];
ry(0.7633546917121068) q[6];
rz(-1.5667451185115147) q[6];
ry(-2.330950657225009) q[7];
rz(-1.5668748366094583) q[7];
ry(-0.8179686746593688) q[8];
rz(1.570795737373322) q[8];
ry(5.791367493124255e-06) q[9];
rz(0.61388838640655) q[9];
ry(0.8169528693940453) q[10];
rz(8.698324601170338e-06) q[10];
ry(-1.5730188907800882) q[11];
rz(-3.106847351021038) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(3.1415388450505497) q[0];
rz(0.9113530495673212) q[0];
ry(-3.1414730825161663) q[1];
rz(-1.0111149695664399) q[1];
ry(-1.5705659433434773) q[2];
rz(-2.0873477251702) q[2];
ry(1.5710398438029323) q[3];
rz(1.5473468993397637) q[3];
ry(1.5703373990914704) q[4];
rz(1.4475370948644004) q[4];
ry(-1.5580559545289328) q[5];
rz(1.9527257646065184) q[5];
ry(1.5710785764516362) q[6];
rz(0.20698666442868952) q[6];
ry(-1.5705233850391584) q[7];
rz(-0.06839069289012399) q[7];
ry(-1.5703346421021473) q[8];
rz(2.965126444011026) q[8];
ry(-2.2984136129316735) q[9];
rz(-1.6122179014028655) q[9];
ry(1.5708507206417435) q[10];
rz(-3.140366536352789) q[10];
ry(-1.6016635491365951) q[11];
rz(0.8446005135201999) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(3.1414494392737504) q[0];
rz(-0.25940443235136534) q[0];
ry(-0.00014381195708956795) q[1];
rz(-2.797338619257748) q[1];
ry(-3.1415870043122696) q[2];
rz(-0.5190959269018718) q[2];
ry(-1.031226827130638e-05) q[3];
rz(0.026030943675876713) q[3];
ry(3.1030946967547374e-07) q[4];
rz(0.19715427083701176) q[4];
ry(-3.141592500605836) q[5];
rz(-2.7012182647573586) q[5];
ry(9.201551484939034e-05) q[6];
rz(-0.20698878124587064) q[6];
ry(3.14149326526684) q[7];
rz(-0.06841606335801177) q[7];
ry(1.761967584027572e-05) q[8];
rz(0.1733118993762197) q[8];
ry(1.5707335337657269) q[9];
rz(-0.42339194393398866) q[9];
ry(1.5708003843796632) q[10];
rz(-0.0057407967318717015) q[10];
ry(1.5705108845755902) q[11];
rz(-2.1890242243878504) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.495790879212598) q[0];
rz(0.016707796099876052) q[0];
ry(1.6458320640771693) q[1];
rz(0.016552402890791327) q[1];
ry(-1.495924329769421) q[2];
rz(0.014375973728040666) q[2];
ry(1.64551449321874) q[3];
rz(0.015568829687454261) q[3];
ry(-1.5664465489946542) q[4];
rz(-1.567347523587783) q[4];
ry(0.06846870415205329) q[5];
rz(-0.05771951069747327) q[5];
ry(1.6365321584831138) q[6];
rz(0.0015353851752406558) q[6];
ry(-1.5761235123054018) q[7];
rz(-3.1401843182497107) q[7];
ry(-1.2648452161193162) q[8];
rz(-3.139830221288581) q[8];
ry(3.138234881259682) q[9];
rz(-1.9932845203891763) q[9];
ry(1.5667308863056197) q[10];
rz(-1.5697935697740635) q[10];
ry(0.007022483961504734) q[11];
rz(0.6192104838421875) q[11];