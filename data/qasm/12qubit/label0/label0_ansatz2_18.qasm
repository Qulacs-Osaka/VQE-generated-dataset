OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-3.139044755666165) q[0];
rz(-1.3968986095191855) q[0];
ry(2.2721269666899944) q[1];
rz(-2.5216380604423074) q[1];
ry(2.0506327155341935) q[2];
rz(-1.3191481815039259) q[2];
ry(2.999584895532363) q[3];
rz(2.15595599888409) q[3];
ry(-1.5042068043375165) q[4];
rz(0.41066849751675427) q[4];
ry(3.089611768434297) q[5];
rz(-1.480945343423809) q[5];
ry(-0.0025843092042165026) q[6];
rz(1.1413736923665825) q[6];
ry(-0.002414001859691872) q[7];
rz(-0.2584305534208778) q[7];
ry(-3.1209613572658905) q[8];
rz(-0.2944956772412519) q[8];
ry(-2.7700257157100117) q[9];
rz(0.2545440047302057) q[9];
ry(-0.155101822015151) q[10];
rz(1.205984110875864) q[10];
ry(3.6827913652892626e-05) q[11];
rz(0.25529309307208214) q[11];
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
ry(-1.5702389201109304) q[0];
rz(-0.00968842505754175) q[0];
ry(-0.35904820152249695) q[1];
rz(-0.7696359058774425) q[1];
ry(1.3699991047310895) q[2];
rz(1.0755022305165758) q[2];
ry(1.3777623619981414) q[3];
rz(-0.04157940385122676) q[3];
ry(3.127779981394362) q[4];
rz(1.4439959692158995) q[4];
ry(-1.7276816748410007) q[5];
rz(1.0500942180281756) q[5];
ry(1.5705865654238544) q[6];
rz(0.6799312471787684) q[6];
ry(1.5688281276096196) q[7];
rz(0.0171116988267149) q[7];
ry(1.544896378496788) q[8];
rz(3.0230517006330153) q[8];
ry(0.14941337429414583) q[9];
rz(-1.5968635373250306) q[9];
ry(-1.5550894185281274) q[10];
rz(-2.898012057771904) q[10];
ry(-1.5703784675519392) q[11];
rz(0.00031911824526265065) q[11];
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
ry(0.5919881077296401) q[0];
rz(-1.5779468492737667) q[0];
ry(-2.1452949094870046) q[1];
rz(0.872200354455726) q[1];
ry(-0.9136923980266474) q[2];
rz(1.9727732818049022) q[2];
ry(2.4946195131861235) q[3];
rz(1.6240713070792576) q[3];
ry(-1.6759080224824077) q[4];
rz(0.5321556962266608) q[4];
ry(0.02726400105593895) q[5];
rz(0.47080302945064123) q[5];
ry(3.1253666729896703) q[6];
rz(-0.8874943165226102) q[6];
ry(0.44320963349691755) q[7];
rz(-1.5785424821124687) q[7];
ry(-0.07415079586211923) q[8];
rz(-1.4575638413414067) q[8];
ry(-1.417009020463567) q[9];
rz(-2.765705416125554) q[9];
ry(-2.8195922863028913) q[10];
rz(3.0701855506720936) q[10];
ry(1.5220124349975583) q[11];
rz(-0.1477286947035426) q[11];
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
ry(2.811362482343103) q[0];
rz(-1.8985280714964912) q[0];
ry(2.8954405965786574) q[1];
rz(1.9327370211531063) q[1];
ry(2.1138183149641216) q[2];
rz(1.354207589019526) q[2];
ry(0.69171158528866) q[3];
rz(0.7732642152134398) q[3];
ry(1.8450294839914498) q[4];
rz(2.157184441371319) q[4];
ry(0.896708136088339) q[5];
rz(1.6611352798101091) q[5];
ry(2.5242231179048904) q[6];
rz(2.113818949776948) q[6];
ry(0.8571247574936365) q[7];
rz(1.57996368604024) q[7];
ry(0.9942238565367241) q[8];
rz(-1.577437497899641) q[8];
ry(2.9071109282884646) q[9];
rz(-0.3399506132823973) q[9];
ry(-0.05742476666964702) q[10];
rz(0.3019283597754167) q[10];
ry(-0.0027871726347013185) q[11];
rz(2.453142025902714) q[11];
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
ry(0.0170389997547824) q[0];
rz(1.887505132161607) q[0];
ry(-0.7335559137111458) q[1];
rz(1.9434163668385798) q[1];
ry(-0.8157117113966936) q[2];
rz(2.87533039391647) q[2];
ry(-0.16585264455400228) q[3];
rz(-0.7719277994042038) q[3];
ry(2.704910485176696) q[4];
rz(-1.7984926332314164) q[4];
ry(2.7812282710072003) q[5];
rz(1.5245848277616278) q[5];
ry(3.130614207160629) q[6];
rz(2.890974355542821) q[6];
ry(-2.858499364421736) q[7];
rz(1.5720646806685554) q[7];
ry(1.4617357249919747) q[8];
rz(-2.8997555061594245) q[8];
ry(-2.2768113003409747) q[9];
rz(-1.590258608823512) q[9];
ry(-0.9596318951206717) q[10];
rz(1.5836796074807438) q[10];
ry(-0.0019323062970488477) q[11];
rz(0.17041926020802659) q[11];
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
ry(2.018059235776734) q[0];
rz(-0.5525449301808976) q[0];
ry(1.2514016273917932) q[1];
rz(1.8988459955855388) q[1];
ry(-1.266331933818793) q[2];
rz(2.2576485640526203) q[2];
ry(-0.9522581368812544) q[3];
rz(1.3624681607629583) q[3];
ry(-1.237892902993928) q[4];
rz(-2.5492886511340815) q[4];
ry(3.029138607859812) q[5];
rz(1.509943394636688) q[5];
ry(-3.12992063623989) q[6];
rz(2.051209274191984) q[6];
ry(0.7737310125294924) q[7];
rz(1.5822270155025002) q[7];
ry(3.129857616163108) q[8];
rz(-2.9074365144129173) q[8];
ry(1.6735534511225316) q[9];
rz(2.7449541504390638) q[9];
ry(0.8221130903302604) q[10];
rz(1.640530587012969) q[10];
ry(-3.1362901586764136) q[11];
rz(-0.684878340873703) q[11];
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
ry(-3.1399606678052265) q[0];
rz(-2.4440305023722324) q[0];
ry(1.2591861313878987) q[1];
rz(1.4304538443222032) q[1];
ry(1.080805914038859) q[2];
rz(-2.2298959604771897) q[2];
ry(-2.5884712508614127) q[3];
rz(1.5979873624985104) q[3];
ry(-2.144864225811377) q[4];
rz(2.207627452331605) q[4];
ry(-1.7660300238001212) q[5];
rz(1.4171275956616) q[5];
ry(0.004446310866485807) q[6];
rz(0.7617542256091127) q[6];
ry(1.2730333662863993) q[7];
rz(1.564939025303894) q[7];
ry(1.1358854564318444) q[8];
rz(1.5616332452487434) q[8];
ry(1.702255328189966) q[9];
rz(-2.6572143498283523) q[9];
ry(2.066225531440925) q[10];
rz(-1.9167750696703862) q[10];
ry(-3.0402064979835295) q[11];
rz(-1.6018461669567243) q[11];
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
ry(-0.0010877076553024168) q[0];
rz(-2.491729413324008) q[0];
ry(-2.8925596114376266) q[1];
rz(2.200339985433264) q[1];
ry(-1.7677667485486124) q[2];
rz(-1.6531403925633832) q[2];
ry(0.3029623457728512) q[3];
rz(1.3159947280029387) q[3];
ry(-0.5834106185512029) q[4];
rz(-0.7373059219619708) q[4];
ry(-0.6378972815098948) q[5];
rz(-1.4162573296518222) q[5];
ry(-3.1354296162638806) q[6];
rz(0.5441142644320075) q[6];
ry(-2.112386485261591) q[7];
rz(1.5845514459131531) q[7];
ry(-0.8885259653451536) q[8];
rz(-1.6617076045220571) q[8];
ry(0.8031629452377763) q[9];
rz(-0.010122057999875091) q[9];
ry(0.15358835854220868) q[10];
rz(1.8270077460389342) q[10];
ry(-0.11531700153016028) q[11];
rz(-1.4676463359563823) q[11];
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
ry(-3.13580698452071) q[0];
rz(2.580302172981029) q[0];
ry(-1.7756584024353208) q[1];
rz(-0.06701919914090036) q[1];
ry(-0.2193943436451489) q[2];
rz(0.13808348866578954) q[2];
ry(1.2787529437075622) q[3];
rz(1.9464624112527797) q[3];
ry(2.394073390612045) q[4];
rz(-1.0708890953438672) q[4];
ry(-1.8844830968429014) q[5];
rz(-2.1353058568040684) q[5];
ry(3.086552814715357) q[6];
rz(1.6437561582044902) q[6];
ry(0.22124484171630954) q[7];
rz(-1.586439019211814) q[7];
ry(0.06788517673356705) q[8];
rz(1.1418716808658658) q[8];
ry(0.6880470430120207) q[9];
rz(0.8138304559796463) q[9];
ry(-0.48249716004179194) q[10];
rz(-1.421380790001879) q[10];
ry(3.1337443031707823) q[11];
rz(-1.4811728755718558) q[11];
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
ry(-3.135138496541729) q[0];
rz(2.2563933826283344) q[0];
ry(1.3802811167340119) q[1];
rz(-1.7879524596269516) q[1];
ry(2.446307397187569) q[2];
rz(-2.726742220240275) q[2];
ry(-2.9621793336771813) q[3];
rz(-1.4404638044074758) q[3];
ry(-2.439236813379824) q[4];
rz(-1.8030850788344832) q[4];
ry(-0.11467380852865894) q[5];
rz(-1.230204059921863) q[5];
ry(-2.5502302618091615) q[6];
rz(1.8582810420292164) q[6];
ry(2.6342076554805716) q[7];
rz(1.6122132796096071) q[7];
ry(3.124080504422231) q[8];
rz(1.021618506694521) q[8];
ry(1.5636093314993713) q[9];
rz(-1.3304288154363126) q[9];
ry(-0.8078676065022935) q[10];
rz(0.09422859013786833) q[10];
ry(1.590302445111897) q[11];
rz(-1.587134611452007) q[11];
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
ry(-1.6867004421477478) q[0];
rz(-1.5718266909116787) q[0];
ry(2.63098859059519) q[1];
rz(1.4596903700860224) q[1];
ry(0.737931180936818) q[2];
rz(-2.6116431249956724) q[2];
ry(-0.2431466767234962) q[3];
rz(-1.2226702771715778) q[3];
ry(-1.2532198433655157) q[4];
rz(2.496806547501381) q[4];
ry(3.0999690004749625) q[5];
rz(1.3865773908212127) q[5];
ry(-0.016698404321020988) q[6];
rz(-1.858534281209943) q[6];
ry(2.889841128596421) q[7];
rz(1.6094314354197545) q[7];
ry(2.416374126448113) q[8];
rz(0.921922896055822) q[8];
ry(-3.0780962024588483) q[9];
rz(3.141137196743143) q[9];
ry(-3.115369863701626) q[10];
rz(1.4607206166503517) q[10];
ry(0.1045306622114186) q[11];
rz(2.9601929320868248) q[11];
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
ry(-2.3059979615955637) q[0];
rz(2.7271598481172745) q[0];
ry(-2.1582920144839055) q[1];
rz(0.4941507384130315) q[1];
ry(0.8904893167922987) q[2];
rz(-0.0031294097664498277) q[2];
ry(-2.371394426343743) q[3];
rz(-1.830551562796063) q[3];
ry(2.1950956014311123) q[4];
rz(-1.9189197033611884) q[4];
ry(-1.2823189100981949) q[5];
rz(1.2729400484325941) q[5];
ry(-1.8915676079274535) q[6];
rz(-1.5718209935879255) q[6];
ry(-0.8207860370773624) q[7];
rz(1.5407974010492695) q[7];
ry(3.1010868588300777) q[8];
rz(0.9441171967575633) q[8];
ry(1.2236709676466353) q[9];
rz(-0.18786801901989403) q[9];
ry(3.1029041719648336) q[10];
rz(-0.21031247002528897) q[10];
ry(0.008920535336366817) q[11];
rz(-2.9603298012232284) q[11];
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
ry(-0.004207683223894405) q[0];
rz(-0.13093549464931992) q[0];
ry(2.5369244109754288) q[1];
rz(0.48089568122438264) q[1];
ry(-2.174050554416078) q[2];
rz(-2.1092525877292507) q[2];
ry(0.28596602282671857) q[3];
rz(-0.814566764821608) q[3];
ry(2.491155264528199) q[4];
rz(-0.14737932435887832) q[4];
ry(0.1637603937300858) q[5];
rz(1.9292689754877808) q[5];
ry(0.9075411005150906) q[6];
rz(-1.5553381733119702) q[6];
ry(2.5973171202216974) q[7];
rz(-1.5975736239326728) q[7];
ry(2.4999681092671464) q[8];
rz(-1.8391908760740248) q[8];
ry(-2.8387770886892) q[9];
rz(1.6263138408006457) q[9];
ry(-0.1507306742347007) q[10];
rz(-1.533939922163007) q[10];
ry(-0.20588753632236748) q[11];
rz(1.5900324084080761) q[11];
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
ry(-3.126656843334739) q[0];
rz(-2.1108415890789143) q[0];
ry(-1.114415812869928) q[1];
rz(-1.3037412611946255) q[1];
ry(-1.1141849049103891) q[2];
rz(-0.3964638203991777) q[2];
ry(0.1227729258831938) q[3];
rz(-1.8816329166968981) q[3];
ry(-2.9126411516823807) q[4];
rz(-0.9194683578121009) q[4];
ry(2.1465906699364052) q[5];
rz(-0.4626431257127129) q[5];
ry(-0.20396594364455023) q[6];
rz(1.7903750961327418) q[6];
ry(-0.674829857334815) q[7];
rz(1.5754520600409183) q[7];
ry(-0.0009843446741851442) q[8];
rz(-0.33033985478458633) q[8];
ry(1.7872984729515806) q[9];
rz(0.7140584534477918) q[9];
ry(-1.087328486451736) q[10];
rz(-2.851818162643461) q[10];
ry(-2.6815184866737276) q[11];
rz(1.5717931016895856) q[11];
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
ry(-0.6841682607748579) q[0];
rz(-1.1051448506789463) q[0];
ry(2.333016081186583) q[1];
rz(1.654726358692138) q[1];
ry(-0.609110907909245) q[2];
rz(1.4480982633993404) q[2];
ry(-1.8863456893411876) q[3];
rz(-1.3655058532437439) q[3];
ry(-2.5430246057887897) q[4];
rz(-0.6617474763576577) q[4];
ry(0.020582649839826623) q[5];
rz(-2.7082983514298693) q[5];
ry(-0.01562636378188742) q[6];
rz(-1.8030129942969677) q[6];
ry(-1.512011623147214) q[7];
rz(1.566601230221183) q[7];
ry(-3.1292545363723727) q[8];
rz(-0.5761561525694007) q[8];
ry(-0.9446916149854543) q[9];
rz(0.2263197190337971) q[9];
ry(-3.1163110153607416) q[10];
rz(0.3096826686528589) q[10];
ry(1.0973770138803332) q[11];
rz(1.4767473685045094) q[11];
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
ry(-3.140997248391458) q[0];
rz(2.338309688637594) q[0];
ry(-0.9161891035846959) q[1];
rz(-2.049099275070731) q[1];
ry(-0.9522984054306702) q[2];
rz(2.891320568809501) q[2];
ry(2.6883128135679026) q[3];
rz(1.2977149148583544) q[3];
ry(2.003036659568274) q[4];
rz(2.5825697153697855) q[4];
ry(2.403504180797668) q[5];
rz(-1.5106184256501576) q[5];
ry(-1.0869142384567045) q[6];
rz(-1.5740952760654974) q[6];
ry(-2.6347540083709893) q[7];
rz(1.569505740096542) q[7];
ry(-1.2214384925114468) q[8];
rz(1.577484170618313) q[8];
ry(2.356085586769882) q[9];
rz(-1.3664723454267058) q[9];
ry(-0.5901094017331625) q[10];
rz(-0.3701699154087849) q[10];
ry(-3.136774722730093) q[11];
rz(-1.6691918113348558) q[11];
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
ry(0.01831151382760865) q[0];
rz(1.6161684616403895) q[0];
ry(1.0034146538533493) q[1];
rz(1.3904690730536986) q[1];
ry(-1.0428378943769883) q[2];
rz(-2.9723076002769195) q[2];
ry(2.928113995988627) q[3];
rz(-1.3303932498362947) q[3];
ry(2.0601221969093197) q[4];
rz(-1.3895140177162073) q[4];
ry(-2.460259991336563) q[5];
rz(1.7172654580571964) q[5];
ry(-0.17474279221002667) q[6];
rz(1.578950731501115) q[6];
ry(-1.7723474124851797) q[7];
rz(-1.4281355263891955) q[7];
ry(-0.6023500160465263) q[8];
rz(-0.41493748765266325) q[8];
ry(1.9809492439599925) q[9];
rz(3.006466574338537) q[9];
ry(-3.1141610975188634) q[10];
rz(-2.1069533807155905) q[10];
ry(-2.159354081816227) q[11];
rz(-1.5720389320544275) q[11];
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
ry(-0.7350895520252427) q[0];
rz(2.106733804419227) q[0];
ry(1.329373424184336) q[1];
rz(2.739753460349799) q[1];
ry(-1.9872048681488534) q[2];
rz(-1.4823866106774126) q[2];
ry(0.03558423963630464) q[3];
rz(-1.9708004491924278) q[3];
ry(-2.2141495025831404) q[4];
rz(-2.87055310192815) q[4];
ry(1.789611316248192) q[5];
rz(3.1000687044279838) q[5];
ry(-1.1185918693839207) q[6];
rz(-1.5742705881093964) q[6];
ry(3.128535535361153) q[7];
rz(1.5702228611802564) q[7];
ry(-2.960654937732723) q[8];
rz(-0.27775661170026) q[8];
ry(-0.9097987213278884) q[9];
rz(-0.8133855399416703) q[9];
ry(0.026618431573394663) q[10];
rz(-2.901765254174246) q[10];
ry(-1.743741200257519) q[11];
rz(-1.5770787671373763) q[11];
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
ry(0.0021096221013504044) q[0];
rz(2.8969705589373826) q[0];
ry(-1.5718615130608278) q[1];
rz(3.097836114236837) q[1];
ry(-0.07849224580746482) q[2];
rz(1.2857396233871574) q[2];
ry(0.06713075291974593) q[3];
rz(-0.9121231480853176) q[3];
ry(0.03378915782276718) q[4];
rz(-1.9239489467576867) q[4];
ry(-3.1354356152243503) q[5];
rz(2.4427731387962184) q[5];
ry(-1.2139538769784277) q[6];
rz(0.30578020131958805) q[6];
ry(0.04943198014822059) q[7];
rz(1.716144468721159) q[7];
ry(-1.5724181978619525) q[8];
rz(-1.6347453881127663) q[8];
ry(-0.07457009224670745) q[9];
rz(1.3968019405615522) q[9];
ry(0.18999074337630883) q[10];
rz(1.487994291923994) q[10];
ry(-2.107488294254114) q[11];
rz(-1.5734159236655254) q[11];
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
ry(-3.128490009885734) q[0];
rz(-2.590081873265334) q[0];
ry(1.5087054932258281) q[1];
rz(0.6432844915558813) q[1];
ry(3.0648750513653553) q[2];
rz(0.34540279301296595) q[2];
ry(0.04373278015047224) q[3];
rz(0.7174638226926926) q[3];
ry(-3.110005862136657) q[4];
rz(2.7419493273452162) q[4];
ry(-3.12010315670108) q[5];
rz(-2.3968345802867517) q[5];
ry(3.133108073680759) q[6];
rz(-1.9084060769999347) q[6];
ry(-1.8356371259102122) q[7];
rz(1.5664305278094934) q[7];
ry(-3.0606188680426776) q[8];
rz(1.508416859272913) q[8];
ry(-1.6277834943680318) q[9];
rz(1.652480244828432) q[9];
ry(-0.8477605883080527) q[10];
rz(2.062875472049884) q[10];
ry(1.0330013716464688) q[11];
rz(1.5687446680516866) q[11];
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
ry(1.6164555129912568) q[0];
rz(1.5551025695908285) q[0];
ry(1.5754035652467415) q[1];
rz(-1.5821837402228507) q[1];
ry(-3.064861263654147) q[2];
rz(-2.235895378322285) q[2];
ry(-0.9805148022675575) q[3];
rz(-1.0703226352164572) q[3];
ry(1.5587613816872041) q[4];
rz(0.045274243509386025) q[4];
ry(-3.077946109914091) q[5];
rz(-1.0090460661315221) q[5];
ry(-0.00018193522357634606) q[6];
rz(-3.0534339956477328) q[6];
ry(0.12625375176968504) q[7];
rz(-1.690524409998737) q[7];
ry(-1.5293406902560915) q[8];
rz(-0.15635625594578226) q[8];
ry(-1.0542964056046848) q[9];
rz(-1.4902025761101925) q[9];
ry(3.115904383184682) q[10];
rz(-1.1874717529751786) q[10];
ry(0.40065926723594475) q[11];
rz(-2.623880695565415) q[11];
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
ry(-1.4038686012522579) q[0];
rz(-1.5719585155566422) q[0];
ry(1.404330098316306) q[1];
rz(-1.560018540495421) q[1];
ry(-2.982883085064883) q[2];
rz(-1.382661694802965) q[2];
ry(-3.0681707897301203) q[3];
rz(2.1144527066271155) q[3];
ry(-1.9406456364697569) q[4];
rz(-0.32026587688807595) q[4];
ry(3.1335932692386597) q[5];
rz(2.2489109151177367) q[5];
ry(-3.121902728451718) q[6];
rz(-2.1204766271756332) q[6];
ry(3.0565629364569897) q[7];
rz(-1.689030344904423) q[7];
ry(-0.007324051989169078) q[8];
rz(0.10982123218201473) q[8];
ry(2.758125434519725) q[9];
rz(-0.1121005096712784) q[9];
ry(-0.18605433407899952) q[10];
rz(1.6634027326857252) q[10];
ry(0.0011185375411926816) q[11];
rz(-0.5152315207211391) q[11];