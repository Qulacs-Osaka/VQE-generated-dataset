OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-0.5998473881609635) q[0];
rz(2.0282517286008876) q[0];
ry(-2.5724608512914955) q[1];
rz(-0.746793762722672) q[1];
ry(-1.633495040299668) q[2];
rz(0.9245931266220326) q[2];
ry(-0.39449978792122486) q[3];
rz(-0.3910556681058176) q[3];
ry(2.884005378735643) q[4];
rz(-2.4623347763380865) q[4];
ry(-0.9172333678160136) q[5];
rz(-1.4667989021164454) q[5];
ry(3.050953333132305) q[6];
rz(-2.407986966732) q[6];
ry(-2.2233735724421084) q[7];
rz(-0.13501887538191226) q[7];
ry(-0.7008529396540504) q[8];
rz(2.0135153618072428) q[8];
ry(-0.7195634414338707) q[9];
rz(-0.7124618040840761) q[9];
ry(2.8903389383339304) q[10];
rz(-2.0558954955971123) q[10];
ry(1.2083451099065838) q[11];
rz(-2.4967456301294066) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-2.409255507029419) q[0];
rz(-1.3675756216343422) q[0];
ry(0.05289342854173196) q[1];
rz(-1.5833312623370475) q[1];
ry(-2.331530911511292) q[2];
rz(2.282117917718246) q[2];
ry(1.586945563295473) q[3];
rz(1.7140530612584834) q[3];
ry(3.0913869392977626) q[4];
rz(-1.3287404025295622) q[4];
ry(-3.1122347726765254) q[5];
rz(-1.9307942760178047) q[5];
ry(2.2033849345673246) q[6];
rz(-2.3902858774149265) q[6];
ry(-0.002956863891161454) q[7];
rz(2.0244029786869793) q[7];
ry(-1.2302980869314626) q[8];
rz(0.30229315990484196) q[8];
ry(3.0460182825246425) q[9];
rz(1.5927550257147232) q[9];
ry(-0.6721203989198496) q[10];
rz(1.462551091769982) q[10];
ry(1.3727075722514803) q[11];
rz(-0.4230506439018944) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.7372443306217578) q[0];
rz(0.18377745648046506) q[0];
ry(-1.2599513910493556) q[1];
rz(1.4866760952821356) q[1];
ry(2.5020201048366437) q[2];
rz(0.22171623089898043) q[2];
ry(0.25782644546003564) q[3];
rz(-1.8606505382051695) q[3];
ry(0.05387594729637955) q[4];
rz(1.2360468127129183) q[4];
ry(2.908753477765868) q[5];
rz(2.064656003849536) q[5];
ry(3.0978012362202665) q[6];
rz(0.5031098953586843) q[6];
ry(-0.9987150270394031) q[7];
rz(2.505935664231934) q[7];
ry(0.08332034457713056) q[8];
rz(-1.4862301365671051) q[8];
ry(1.2736493093317138) q[9];
rz(-2.2131587222838127) q[9];
ry(0.412061715257944) q[10];
rz(2.2139124601857607) q[10];
ry(-1.576685720979836) q[11];
rz(-0.08942126205590915) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.0724029085217763) q[0];
rz(-0.06304638866996902) q[0];
ry(0.9060570940435092) q[1];
rz(-1.1743415000686737) q[1];
ry(2.1832687190743485) q[2];
rz(-1.973481150013478) q[2];
ry(0.2777206245676576) q[3];
rz(0.17049898852476805) q[3];
ry(3.038096790830094) q[4];
rz(-1.839111949860986) q[4];
ry(-3.1301163736030335) q[5];
rz(2.168782723717789) q[5];
ry(0.8758468088841703) q[6];
rz(0.5877828815927302) q[6];
ry(-3.13738728492546) q[7];
rz(1.9803076976868976) q[7];
ry(-0.5334954608563578) q[8];
rz(-3.02056883256393) q[8];
ry(-0.25195711661244863) q[9];
rz(2.8700155934995135) q[9];
ry(1.1800973798501975) q[10];
rz(1.9447220648579693) q[10];
ry(0.1424579878558845) q[11];
rz(1.2773183688420857) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.827634398433125) q[0];
rz(-1.5298886930281466) q[0];
ry(0.19134615390976073) q[1];
rz(0.6212395747588813) q[1];
ry(1.2562460884179216) q[2];
rz(0.09687538155820086) q[2];
ry(-2.8385673106665146) q[3];
rz(0.5955703302798359) q[3];
ry(-1.3704631488531271) q[4];
rz(-1.3803114243562842) q[4];
ry(-0.027799106726644318) q[5];
rz(-3.0860472521958573) q[5];
ry(0.1652539401159663) q[6];
rz(-2.5294144825902967) q[6];
ry(0.5844356792898467) q[7];
rz(2.6600257041234574) q[7];
ry(-0.1000080642356902) q[8];
rz(-0.07182089594512464) q[8];
ry(0.5410103713105423) q[9];
rz(0.08124103091566656) q[9];
ry(-2.2602442516435706) q[10];
rz(-2.5162161399820597) q[10];
ry(3.109891118958699) q[11];
rz(0.4835382173143236) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.2986963398342721) q[0];
rz(-0.8444069877535073) q[0];
ry(-2.3300827297902793) q[1];
rz(2.5373895153476433) q[1];
ry(-2.993426146163773) q[2];
rz(0.8680794794868474) q[2];
ry(-2.1201069272123334) q[3];
rz(0.15309701855546243) q[3];
ry(0.004253232511632454) q[4];
rz(1.0526611412381288) q[4];
ry(-3.0378141419435867) q[5];
rz(2.9678642983190473) q[5];
ry(-0.0135872843016741) q[6];
rz(2.736186740490821) q[6];
ry(3.130182376936497) q[7];
rz(-2.285992853940884) q[7];
ry(-2.876711336202132) q[8];
rz(0.22326701084239797) q[8];
ry(-2.889002873575296) q[9];
rz(0.6577689458429439) q[9];
ry(-0.0034307026503217486) q[10];
rz(1.777318315088924) q[10];
ry(1.5118245276170648) q[11];
rz(2.782369064403418) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-2.790586985191218) q[0];
rz(-0.7203435826292796) q[0];
ry(2.4624444778999677) q[1];
rz(-2.403418711359163) q[1];
ry(3.0878237980512395) q[2];
rz(0.056290052616858155) q[2];
ry(0.6783311005986654) q[3];
rz(-1.487907334033202) q[3];
ry(1.203005943105264) q[4];
rz(-1.813093668594652) q[4];
ry(0.48802291801487835) q[5];
rz(3.0009918660230563) q[5];
ry(-0.15488894067850298) q[6];
rz(-1.9605126745592094) q[6];
ry(-1.5399967268953696) q[7];
rz(-2.1301318521371906) q[7];
ry(-0.09644350587349315) q[8];
rz(-1.5889605773527766) q[8];
ry(1.6286344059121327) q[9];
rz(-1.120853980622053) q[9];
ry(-0.3160390835511748) q[10];
rz(-2.5697583307415033) q[10];
ry(-0.8768587216192749) q[11];
rz(1.4196514173559975) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-2.902021471233135) q[0];
rz(-0.4216701572352628) q[0];
ry(-1.8192490494288842) q[1];
rz(1.1966622435442364) q[1];
ry(-2.7485198798700248) q[2];
rz(-2.2543518685178263) q[2];
ry(-2.295869923422094) q[3];
rz(-1.8698492468386627) q[3];
ry(0.04452079429576075) q[4];
rz(1.7985180249489372) q[4];
ry(-0.005859957941617644) q[5];
rz(2.620581298420584) q[5];
ry(-3.0891319671747834) q[6];
rz(-0.7112255266194545) q[6];
ry(-3.1414543617148594) q[7];
rz(-0.047318507297216435) q[7];
ry(1.513649249684436) q[8];
rz(-0.7237544970643432) q[8];
ry(1.387291970501484) q[9];
rz(-0.04856566821503965) q[9];
ry(1.9763294354103285) q[10];
rz(0.7088041079830366) q[10];
ry(-1.1389838119243205) q[11];
rz(2.6313240883620153) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.4729911925591548) q[0];
rz(-0.6909250000961344) q[0];
ry(-2.2152309347984547) q[1];
rz(1.8432278806838873) q[1];
ry(2.764939561453384) q[2];
rz(2.309246657120484) q[2];
ry(-0.44784975647974884) q[3];
rz(-1.0373213611763212) q[3];
ry(1.7455717203802343) q[4];
rz(-2.0086039864758463) q[4];
ry(-2.956970449679806) q[5];
rz(1.6497560550575987) q[5];
ry(0.030396847331211955) q[6];
rz(-2.408845686678504) q[6];
ry(-0.07315736331851053) q[7];
rz(2.582296245759651) q[7];
ry(0.07806787396852205) q[8];
rz(0.02584935390963587) q[8];
ry(-1.0691345558816536) q[9];
rz(3.080005164930009) q[9];
ry(-1.1672493781899576) q[10];
rz(2.765026243360602) q[10];
ry(2.9235781993024523) q[11];
rz(2.226958457587744) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-0.8877175813035313) q[0];
rz(-0.3281722385283653) q[0];
ry(2.879977750763347) q[1];
rz(1.819773467094432) q[1];
ry(1.4033437699823894) q[2];
rz(3.015002978260611) q[2];
ry(2.863853417465821) q[3];
rz(1.9640165205747737) q[3];
ry(0.0049586269170909915) q[4];
rz(-1.3864945067899965) q[4];
ry(1.3138015659386777) q[5];
rz(-0.230907868390247) q[5];
ry(0.1365400246984274) q[6];
rz(2.6361299587497378) q[6];
ry(-2.411186616532082) q[7];
rz(-0.9378820369339937) q[7];
ry(1.4327859427798544) q[8];
rz(1.9028609776264975) q[8];
ry(1.6751166634239656) q[9];
rz(-1.8859119889700733) q[9];
ry(0.998414855526054) q[10];
rz(-0.27340444585659307) q[10];
ry(-2.2135746990754406) q[11];
rz(-1.6520903010837238) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(0.4737866834354684) q[0];
rz(1.7704746775356197) q[0];
ry(2.059870066858161) q[1];
rz(2.2765388706448277) q[1];
ry(-1.5123463854772599) q[2];
rz(-1.4140241051797084) q[2];
ry(3.105102902999044) q[3];
rz(1.6070847442986878) q[3];
ry(-0.03844408611147948) q[4];
rz(3.016794390617392) q[4];
ry(-0.00012826371596473177) q[5];
rz(-2.752136532462716) q[5];
ry(0.0005358886376880534) q[6];
rz(2.456333921072526) q[6];
ry(0.015593301290648459) q[7];
rz(0.9422459337101641) q[7];
ry(-2.148981985256167) q[8];
rz(0.6880628034226489) q[8];
ry(0.4192988281743322) q[9];
rz(-2.0159063133255914) q[9];
ry(-1.0244159914378028) q[10];
rz(0.9484349659846795) q[10];
ry(-0.927470083682584) q[11];
rz(0.058834022810248854) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(0.8291901922172666) q[0];
rz(1.047915243023069) q[0];
ry(-2.661392491225774) q[1];
rz(-0.1677490307702012) q[1];
ry(0.7697487758165593) q[2];
rz(-0.27733579834438254) q[2];
ry(2.591986355827796) q[3];
rz(-1.7647758482593654) q[3];
ry(3.1265958710224604) q[4];
rz(-0.9875518662068253) q[4];
ry(-1.1110286544805223) q[5];
rz(-1.9773901758279095) q[5];
ry(3.138461855776698) q[6];
rz(1.3956032500602862) q[6];
ry(-0.7309028899814294) q[7];
rz(-1.3075242196849644) q[7];
ry(-0.009558545712226696) q[8];
rz(3.053731166918704) q[8];
ry(2.5868788775970972) q[9];
rz(0.35858980392908996) q[9];
ry(0.8135091692097198) q[10];
rz(1.391250572750835) q[10];
ry(-1.8280571933472824) q[11];
rz(0.47092405552817757) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.4661247690633565) q[0];
rz(-1.3346885824420558) q[0];
ry(1.432196342622131) q[1];
rz(-1.599232493175009) q[1];
ry(-1.4933676717602182) q[2];
rz(-2.7383504448726317) q[2];
ry(0.41854957338493115) q[3];
rz(3.06450694150688) q[3];
ry(0.18988849075891248) q[4];
rz(-0.23465881374888312) q[4];
ry(2.9283981721822827) q[5];
rz(3.0281972121938576) q[5];
ry(-2.8410553741880125) q[6];
rz(-2.044590426880627) q[6];
ry(-1.3965996376856635) q[7];
rz(1.6589798037241508) q[7];
ry(1.700743732665143) q[8];
rz(2.9902832887849855) q[8];
ry(0.18355408720034738) q[9];
rz(-0.6240355354625918) q[9];
ry(-0.38365860958897263) q[10];
rz(1.9255150599572821) q[10];
ry(-2.6943141883963917) q[11];
rz(0.0213152787919002) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.5128081622493834) q[0];
rz(-2.6028916536336224) q[0];
ry(1.2071949094427887) q[1];
rz(-2.1901312075197183) q[1];
ry(0.3088630378103397) q[2];
rz(1.3559260013897556) q[2];
ry(1.4842213329066167) q[3];
rz(-0.4452958351717964) q[3];
ry(3.124160174900056) q[4];
rz(1.9938308471455126) q[4];
ry(0.07429863399206393) q[5];
rz(0.15772293578778593) q[5];
ry(-0.030362158711995768) q[6];
rz(-1.9677986154986522) q[6];
ry(3.136402441110012) q[7];
rz(1.5852197296861446) q[7];
ry(-0.01761745059087172) q[8];
rz(-2.303879704564805) q[8];
ry(-0.5034531337489696) q[9];
rz(-0.6592704821882575) q[9];
ry(-1.142567477950104) q[10];
rz(2.947959527246761) q[10];
ry(1.9485279644539215) q[11];
rz(-2.394077000347047) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.7261913441105359) q[0];
rz(0.35142634103938786) q[0];
ry(-1.993928262061397) q[1];
rz(2.88732341828344) q[1];
ry(1.402029064498488) q[2];
rz(3.05951551294395) q[2];
ry(0.031565189347459034) q[3];
rz(1.8049625193272087) q[3];
ry(-0.6812522674775424) q[4];
rz(2.997649186107589) q[4];
ry(-0.8458641608029938) q[5];
rz(-1.1972708154586602) q[5];
ry(2.4838510585635967) q[6];
rz(0.004434500326770917) q[6];
ry(1.9143917521799168) q[7];
rz(0.5488678138935388) q[7];
ry(-2.8416987529758755) q[8];
rz(-0.7680404369272683) q[8];
ry(0.8551368796010257) q[9];
rz(0.29880716195291884) q[9];
ry(-0.10469637389960813) q[10];
rz(-2.6207632978960267) q[10];
ry(-1.5530542320763938) q[11];
rz(0.11947586979115085) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-2.5137751952968417) q[0];
rz(-0.16694906086538422) q[0];
ry(0.3149951919868533) q[1];
rz(-0.9691980108687076) q[1];
ry(2.0632240532546233) q[2];
rz(2.324959208813638) q[2];
ry(-0.07757236658641363) q[3];
rz(-0.9799816057478762) q[3];
ry(3.119962940999122) q[4];
rz(2.6691526706470152) q[4];
ry(0.0009922411196612302) q[5];
rz(1.4083549606746708) q[5];
ry(3.1384606472092687) q[6];
rz(0.23733212433146547) q[6];
ry(0.0006131972136644848) q[7];
rz(-1.7840367815782472) q[7];
ry(-3.107435803658749) q[8];
rz(-2.427886416876956) q[8];
ry(-1.784719540263196) q[9];
rz(-2.145358583488408) q[9];
ry(-1.4866691549253552) q[10];
rz(-0.024273347830782832) q[10];
ry(-1.5409282199787275) q[11];
rz(-0.6421729312230383) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(2.9376140263352) q[0];
rz(-0.617871838081796) q[0];
ry(-2.281213360272513) q[1];
rz(1.9925669742390733) q[1];
ry(-2.8070385144345416) q[2];
rz(-0.8447171126453153) q[2];
ry(-0.14049966925664936) q[3];
rz(-2.985954746305763) q[3];
ry(-0.5354524546586328) q[4];
rz(1.5318628978263398) q[4];
ry(-1.1009490715231725) q[5];
rz(-0.040612320124225144) q[5];
ry(0.5227678269081251) q[6];
rz(1.7727069478749211) q[6];
ry(-0.16849813827872492) q[7];
rz(1.2309990583412798) q[7];
ry(2.908424355288274) q[8];
rz(2.2708875127903525) q[8];
ry(3.0596149253133387) q[9];
rz(1.0043680513794646) q[9];
ry(1.7838619809780647) q[10];
rz(3.092524742563674) q[10];
ry(-3.077847765978014) q[11];
rz(1.2886264961511724) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.8674996356708649) q[0];
rz(-0.23919609893063667) q[0];
ry(-1.9042873564882843) q[1];
rz(-0.2738023713315414) q[1];
ry(0.597512108653675) q[2];
rz(1.6133941472823354) q[2];
ry(0.09430922399640362) q[3];
rz(2.6444070567872697) q[3];
ry(-3.1087272359299147) q[4];
rz(-0.16166790683470042) q[4];
ry(-3.1311833980313155) q[5];
rz(2.490530880150792) q[5];
ry(0.0541489605664951) q[6];
rz(0.024867822690791463) q[6];
ry(-3.141147061448188) q[7];
rz(-1.5588936159066797) q[7];
ry(0.013571229983755373) q[8];
rz(-2.098384141968878) q[8];
ry(-1.7327699652878388) q[9];
rz(-1.7055325243279558) q[9];
ry(-1.4094301917900793) q[10];
rz(-3.0562662174684387) q[10];
ry(1.5470840352571997) q[11];
rz(-2.91487707848126) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.894341664486013) q[0];
rz(-2.6371913044092583) q[0];
ry(3.0784841672088046) q[1];
rz(-0.3437220108000467) q[1];
ry(2.6985925524873595) q[2];
rz(1.3135626703612913) q[2];
ry(1.7253456068030237) q[3];
rz(-2.007812312795767) q[3];
ry(1.6720530809483576) q[4];
rz(-0.9527326629887044) q[4];
ry(-0.2517676723131501) q[5];
rz(1.7962762270024157) q[5];
ry(-1.5983058872050275) q[6];
rz(-0.7985431381120768) q[6];
ry(1.6347662042260365) q[7];
rz(0.12693535446559331) q[7];
ry(1.5241232715984854) q[8];
rz(0.1402986738282186) q[8];
ry(-2.907682452911521) q[9];
rz(-1.5584213088250183) q[9];
ry(-1.4042311089285902) q[10];
rz(-0.10935468832773215) q[10];
ry(3.0540445876830176) q[11];
rz(-1.3192759540534968) q[11];