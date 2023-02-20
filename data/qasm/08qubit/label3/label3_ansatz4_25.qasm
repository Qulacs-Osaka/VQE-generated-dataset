OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-0.710344622115343) q[0];
rz(2.3425882898520625) q[0];
ry(-0.23623447076009985) q[1];
rz(-1.4699505193474427) q[1];
ry(2.7774329390259944) q[2];
rz(-1.9168825979339144) q[2];
ry(2.0244875269861264) q[3];
rz(0.9651547034590182) q[3];
ry(-1.9264369571418154) q[4];
rz(-0.16966223492242413) q[4];
ry(1.4817530629448634) q[5];
rz(1.5590392830290032) q[5];
ry(1.4160114935875932) q[6];
rz(-1.2807108518032777) q[6];
ry(0.8799182681719477) q[7];
rz(-1.4524776417248768) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.2052317931438008) q[0];
rz(-0.1416050365914776) q[0];
ry(3.018855926672174) q[1];
rz(1.5344989823927047) q[1];
ry(0.35492204459939425) q[2];
rz(3.0422633949865805) q[2];
ry(1.772295422082151) q[3];
rz(2.8791216257004866) q[3];
ry(1.716296393254653) q[4];
rz(-0.676248990053838) q[4];
ry(0.19927731195405393) q[5];
rz(1.6772277998841474) q[5];
ry(-1.2746911504493328) q[6];
rz(2.199647281356422) q[6];
ry(-2.052055316972077) q[7];
rz(0.19504353286393736) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.4841873051715333) q[0];
rz(1.7096150406923618) q[0];
ry(-2.9990251500655374) q[1];
rz(1.2746060607683196) q[1];
ry(0.6885532113769735) q[2];
rz(-1.456979562320717) q[2];
ry(1.15004105072017) q[3];
rz(-1.1303530837501272) q[3];
ry(-2.9089215700196656) q[4];
rz(3.0735699344906457) q[4];
ry(1.0290186836636475) q[5];
rz(-1.9099825938689325) q[5];
ry(-2.33420834473296) q[6];
rz(-0.5945016450819921) q[6];
ry(-0.8075268017318273) q[7];
rz(3.1178234733149894) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.14339050904916031) q[0];
rz(-1.8053393427648086) q[0];
ry(-1.480829326853474) q[1];
rz(0.00906948451826619) q[1];
ry(-0.34483996778580545) q[2];
rz(1.53162966009383) q[2];
ry(-0.9456693970670349) q[3];
rz(-1.0838062279117555) q[3];
ry(-0.7505823454522753) q[4];
rz(-1.6673818764224348) q[4];
ry(-1.8970349889652018) q[5];
rz(3.072552956107749) q[5];
ry(1.6179775422659644) q[6];
rz(-0.21412355846770928) q[6];
ry(0.7081038759066477) q[7];
rz(2.351049459702498) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.5168790467426487) q[0];
rz(0.20606880676124897) q[0];
ry(-0.8901284339150921) q[1];
rz(-1.2092275067072211) q[1];
ry(1.3958257446139632) q[2];
rz(2.305283797740287) q[2];
ry(2.253054032605484) q[3];
rz(2.2111233486567636) q[3];
ry(-0.9640517064763855) q[4];
rz(-0.6854182816232883) q[4];
ry(-0.5037527599164973) q[5];
rz(-1.44005874740309) q[5];
ry(-1.5351189365456364) q[6];
rz(-2.798811518378561) q[6];
ry(1.82346098748946) q[7];
rz(0.11974090438284056) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.3960013943897764) q[0];
rz(2.3623404404714843) q[0];
ry(-0.48756028202787643) q[1];
rz(1.1508838850490006) q[1];
ry(-2.389093547448543) q[2];
rz(1.6448672287921042) q[2];
ry(1.4332836675376617) q[3];
rz(-2.674383646132885) q[3];
ry(1.1434011121648453) q[4];
rz(-2.06394966585494) q[4];
ry(2.1271385567537653) q[5];
rz(0.7079716255905543) q[5];
ry(-2.646063254840285) q[6];
rz(1.2170609774961647) q[6];
ry(2.706918522845391) q[7];
rz(-0.4145816987175965) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.997211355092037) q[0];
rz(-1.9804400117293386) q[0];
ry(-1.3844508302118217) q[1];
rz(1.9857269925677166) q[1];
ry(-0.7358090386148932) q[2];
rz(-1.8744450557288665) q[2];
ry(2.4258136433920456) q[3];
rz(3.0765645137176367) q[3];
ry(0.9225125152586986) q[4];
rz(1.3614494492958882) q[4];
ry(1.7376606499783012) q[5];
rz(0.10398035674381578) q[5];
ry(-2.6924675709010506) q[6];
rz(-2.239029436419882) q[6];
ry(0.23529506346587523) q[7];
rz(2.137263322487304) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.6557230519003276) q[0];
rz(2.103040779251345) q[0];
ry(-1.2405142711133121) q[1];
rz(-3.078388982006861) q[1];
ry(-0.26321352766037626) q[2];
rz(0.05422607758935193) q[2];
ry(-0.7293786580263646) q[3];
rz(1.549584763046557) q[3];
ry(-0.5606719116570339) q[4];
rz(2.9206515276463545) q[4];
ry(2.2434131780761364) q[5];
rz(-0.2686347067799284) q[5];
ry(-0.4510453633644702) q[6];
rz(1.0769391224854912) q[6];
ry(-3.0105309122169333) q[7];
rz(1.134739118179404) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.1899588947618147) q[0];
rz(-0.5244621561499336) q[0];
ry(2.7994799704542683) q[1];
rz(0.878775587164215) q[1];
ry(-2.763965515477676) q[2];
rz(1.0833270865786795) q[2];
ry(1.210434054767802) q[3];
rz(-0.6198614779789766) q[3];
ry(2.69588422410091) q[4];
rz(2.2347852528921566) q[4];
ry(0.9733730053300719) q[5];
rz(2.8152670879228725) q[5];
ry(-2.894292256434834) q[6];
rz(-2.8345775045805834) q[6];
ry(2.3294159982569407) q[7];
rz(-1.6147462664157415) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.46100603672131957) q[0];
rz(-1.2591785359584355) q[0];
ry(-1.9020009320996722) q[1];
rz(1.2551199719991004) q[1];
ry(2.5117263153485707) q[2];
rz(0.12156367493927966) q[2];
ry(-1.813061378531813) q[3];
rz(1.5900699742555526) q[3];
ry(0.2779993685595074) q[4];
rz(-3.069688021027755) q[4];
ry(0.8710382752238454) q[5];
rz(-0.41663002913066194) q[5];
ry(-1.5956903755149912) q[6];
rz(1.9923969196962155) q[6];
ry(0.6278918705041718) q[7];
rz(-0.9713538825791335) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.3798522525912083) q[0];
rz(2.747732305589205) q[0];
ry(1.355867645413996) q[1];
rz(-0.09014592852602821) q[1];
ry(0.5730197049173498) q[2];
rz(0.6461036596790395) q[2];
ry(0.036774184078597116) q[3];
rz(-2.836997926770135) q[3];
ry(-2.8526237625672337) q[4];
rz(-2.145821372532388) q[4];
ry(-2.1893919477252712) q[5];
rz(-0.12004402065689089) q[5];
ry(-0.9885298975270089) q[6];
rz(-3.052748602566154) q[6];
ry(-1.879526436193835) q[7];
rz(0.9393952774473484) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.6410754428895008) q[0];
rz(1.5221751854769539) q[0];
ry(-2.0399956253196043) q[1];
rz(2.7359054418033866) q[1];
ry(0.6507506446593793) q[2];
rz(-1.0432703449852816) q[2];
ry(-0.47658047783374546) q[3];
rz(-0.7622357430030613) q[3];
ry(1.3981511848677997) q[4];
rz(-1.6046533046585507) q[4];
ry(-0.8791367161143224) q[5];
rz(-2.1587990939085433) q[5];
ry(2.195768208428902) q[6];
rz(1.8838390932693905) q[6];
ry(-1.5734646982666065) q[7];
rz(1.3157393345692237) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.9342040369434637) q[0];
rz(-0.495950695757779) q[0];
ry(2.5479558840269076) q[1];
rz(-1.179652475560414) q[1];
ry(-0.39109157878884265) q[2];
rz(1.0712302797952533) q[2];
ry(1.4209910747624779) q[3];
rz(0.1152481093222546) q[3];
ry(-1.4606204203483268) q[4];
rz(-0.30088226214226366) q[4];
ry(0.07779948110871518) q[5];
rz(1.0273279033170497) q[5];
ry(-0.6523036213532291) q[6];
rz(-0.6110496764316967) q[6];
ry(0.7910249949302429) q[7];
rz(2.7567539461735584) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.3305510133450444) q[0];
rz(1.9150931024374647) q[0];
ry(-2.5842433782912257) q[1];
rz(-1.529643988238357) q[1];
ry(1.6379485092238049) q[2];
rz(-2.354464444383968) q[2];
ry(-2.30897509864503) q[3];
rz(-2.8194488587445794) q[3];
ry(1.6450214293366008) q[4];
rz(0.7406263697533415) q[4];
ry(-0.4668592623875272) q[5];
rz(-2.713316551958349) q[5];
ry(0.5021532411028371) q[6];
rz(-1.2465934222006965) q[6];
ry(0.39548096595067866) q[7];
rz(-2.42504796762115) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.3139184321975674) q[0];
rz(2.0781558528364883) q[0];
ry(-1.162330128167177) q[1];
rz(2.2245768410033477) q[1];
ry(-1.1605182284535045) q[2];
rz(-2.5173721058391223) q[2];
ry(1.787115713118828) q[3];
rz(2.431529792738766) q[3];
ry(0.30052622741564217) q[4];
rz(-1.905755619710165) q[4];
ry(0.33228899861555045) q[5];
rz(3.127446112084889) q[5];
ry(1.5674758866271583) q[6];
rz(-2.7857413934048294) q[6];
ry(-1.601849593228963) q[7];
rz(-2.3092737381420947) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-3.1080530946063134) q[0];
rz(-2.561949119328989) q[0];
ry(0.6680924575935909) q[1];
rz(0.3601691327772736) q[1];
ry(-1.358035153976033) q[2];
rz(-1.9595787960879856) q[2];
ry(-0.9832549422300191) q[3];
rz(2.03390917304238) q[3];
ry(-0.2592263638985733) q[4];
rz(2.567457262491717) q[4];
ry(0.5284442046847867) q[5];
rz(2.4165157996961613) q[5];
ry(2.2657811366438843) q[6];
rz(1.8266108287704856) q[6];
ry(-1.4468777023678319) q[7];
rz(2.2680369464215397) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.2685126477285384) q[0];
rz(-1.8120935891834495) q[0];
ry(-2.8605347037161843) q[1];
rz(-0.23584877265783957) q[1];
ry(-2.3735890155081374) q[2];
rz(-2.491058434274489) q[2];
ry(-2.5682417863458396) q[3];
rz(3.1170953251306517) q[3];
ry(2.4422441752486104) q[4];
rz(2.9320586035921843) q[4];
ry(-1.0039895335980562) q[5];
rz(1.948155097059889) q[5];
ry(-2.1820367483060363) q[6];
rz(2.58221030661185) q[6];
ry(-1.703614108370461) q[7];
rz(-1.1249322030331603) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.810270239984058) q[0];
rz(3.095767851458001) q[0];
ry(-2.114309791123893) q[1];
rz(-0.8513473899073855) q[1];
ry(1.6695735833454455) q[2];
rz(-1.0709739807444123) q[2];
ry(-0.8887878714865751) q[3];
rz(-3.043552864508904) q[3];
ry(-1.4685917725136495) q[4];
rz(-2.5009816323981537) q[4];
ry(-1.1235154311171571) q[5];
rz(-3.1111963380933343) q[5];
ry(-1.2160144627837133) q[6];
rz(-0.39831064368466423) q[6];
ry(-0.8278878092303584) q[7];
rz(1.8932129787369603) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.6529323648494447) q[0];
rz(1.2141711321754274) q[0];
ry(1.80513930553943) q[1];
rz(3.0183742297436655) q[1];
ry(0.32695588452069196) q[2];
rz(-0.8744456326164318) q[2];
ry(-1.0634366421012824) q[3];
rz(1.5082744559311252) q[3];
ry(1.569929438615052) q[4];
rz(-0.13626573272367848) q[4];
ry(-0.7099136998313157) q[5];
rz(0.6643160471967331) q[5];
ry(2.046193673193436) q[6];
rz(0.7339523399304136) q[6];
ry(-2.012098847630263) q[7];
rz(1.9755281796899318) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.50377849570369) q[0];
rz(-0.5505001952685853) q[0];
ry(-0.9289006103548504) q[1];
rz(0.41817466643736617) q[1];
ry(1.9703520188679102) q[2];
rz(-2.482370629598022) q[2];
ry(-0.18468332088840017) q[3];
rz(2.220393967827752) q[3];
ry(2.306631288088065) q[4];
rz(1.2393973851874018) q[4];
ry(1.1273601323345546) q[5];
rz(-2.37266155902977) q[5];
ry(-1.5384799857300693) q[6];
rz(-0.18419388894178823) q[6];
ry(-0.8682050692957424) q[7];
rz(0.3602807783459157) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.3011501498855638) q[0];
rz(2.7743154535215693) q[0];
ry(2.470453777109624) q[1];
rz(1.1069228573044663) q[1];
ry(-1.487029250338418) q[2];
rz(-1.9962595953480307) q[2];
ry(-0.6750742359983076) q[3];
rz(-2.825947136159606) q[3];
ry(-1.2002349929729155) q[4];
rz(-2.2708674056917646) q[4];
ry(-0.8958164246767595) q[5];
rz(2.386427435564644) q[5];
ry(1.7769999961871434) q[6];
rz(1.2095450424219125) q[6];
ry(1.0508314704761108) q[7];
rz(-2.1332906980153497) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.5182053079548528) q[0];
rz(1.775757069185649) q[0];
ry(-2.8907621813832463) q[1];
rz(2.299688374602259) q[1];
ry(1.8075768144317008) q[2];
rz(1.3187991753071282) q[2];
ry(-2.8688761842580903) q[3];
rz(1.1538068336343958) q[3];
ry(1.4667205743729232) q[4];
rz(-0.029280224139194806) q[4];
ry(1.9803664945133344) q[5];
rz(-2.80232445452118) q[5];
ry(0.36504366560597723) q[6];
rz(1.576922552042583) q[6];
ry(-0.18157462278482678) q[7];
rz(2.720844395791116) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.9238324374968663) q[0];
rz(-0.31635650649678604) q[0];
ry(-2.976647043599516) q[1];
rz(-1.2434623535592515) q[1];
ry(-0.6573918317521583) q[2];
rz(-0.05733992268939048) q[2];
ry(-2.78018606522021) q[3];
rz(2.26905307866987) q[3];
ry(2.10809089288413) q[4];
rz(-0.4477899726450824) q[4];
ry(0.29139327450462904) q[5];
rz(-1.5954827044574538) q[5];
ry(-0.7562560556898763) q[6];
rz(-1.446299413905378) q[6];
ry(-2.283069590988555) q[7];
rz(2.2065928120863063) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.457018207150193) q[0];
rz(0.18946738716015515) q[0];
ry(-1.811866230002701) q[1];
rz(-1.7712133711132516) q[1];
ry(-1.6973111156693228) q[2];
rz(1.6402934883042273) q[2];
ry(0.7685219994466037) q[3];
rz(-1.4574777439188473) q[3];
ry(0.5984716710250656) q[4];
rz(3.1131537047282407) q[4];
ry(-2.811393498708194) q[5];
rz(2.3996715399761537) q[5];
ry(2.8159244973623743) q[6];
rz(1.7499301227414037) q[6];
ry(-1.0279630226321388) q[7];
rz(0.12324875926270361) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.9810845052456967) q[0];
rz(-0.02262819074424005) q[0];
ry(-1.1308761757939303) q[1];
rz(-2.7295372955740937) q[1];
ry(-0.8316544110510533) q[2];
rz(0.9616734700495186) q[2];
ry(-1.0154723977218731) q[3];
rz(1.7651798913375591) q[3];
ry(-1.5615325829122302) q[4];
rz(-0.9467097690623083) q[4];
ry(2.2673054468490568) q[5];
rz(-1.199720278346807) q[5];
ry(-2.9745645810285017) q[6];
rz(0.7491885518415687) q[6];
ry(0.12255830308195946) q[7];
rz(0.8220367256841551) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-3.0075275732447913) q[0];
rz(0.5557905801076748) q[0];
ry(-0.19270809312359494) q[1];
rz(-1.5949207382859518) q[1];
ry(-1.6830994530108392) q[2];
rz(2.654036071628231) q[2];
ry(0.9726871743728639) q[3];
rz(-0.7274579451001065) q[3];
ry(1.4459498520864091) q[4];
rz(0.3372059164572167) q[4];
ry(1.7479097660424072) q[5];
rz(1.80070206976312) q[5];
ry(-1.68442846721843) q[6];
rz(-3.0783267765668416) q[6];
ry(1.0105414857100774) q[7];
rz(2.478408072816569) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.5225684100570662) q[0];
rz(2.743867026121899) q[0];
ry(-0.3364388362741951) q[1];
rz(-0.21477046850547177) q[1];
ry(2.513944500405755) q[2];
rz(2.358207281226518) q[2];
ry(0.9729697280255765) q[3];
rz(-1.1982112101536093) q[3];
ry(-1.904477481461667) q[4];
rz(0.680006505393207) q[4];
ry(-1.6856349021181911) q[5];
rz(-0.5796086816046027) q[5];
ry(-0.341575426961656) q[6];
rz(2.8719534987570596) q[6];
ry(-0.5484283201075126) q[7];
rz(1.653499559902711) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.1510781339822143) q[0];
rz(-3.066833822518722) q[0];
ry(1.4846848093536593) q[1];
rz(0.520225635952122) q[1];
ry(1.6306360400981892) q[2];
rz(-2.288948714462367) q[2];
ry(0.21554368073815947) q[3];
rz(-0.5371160410619664) q[3];
ry(2.1577675789486275) q[4];
rz(-0.4660452384133818) q[4];
ry(-2.3290384621573277) q[5];
rz(-2.731433651185385) q[5];
ry(-0.17797199157738586) q[6];
rz(2.9689659838894356) q[6];
ry(1.2724653236825283) q[7];
rz(-2.2359063712347345) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.8418761719058213) q[0];
rz(0.23289437598458385) q[0];
ry(2.836378306186383) q[1];
rz(0.38071914518810596) q[1];
ry(-1.678199463305468) q[2];
rz(2.855749609592683) q[2];
ry(0.6082656491979961) q[3];
rz(-2.3140775256144184) q[3];
ry(2.0629241694955516) q[4];
rz(-2.9651516400234232) q[4];
ry(-3.013175158864349) q[5];
rz(-1.5508496248533508) q[5];
ry(-0.44814608804195416) q[6];
rz(-2.464948002474013) q[6];
ry(2.8949565107660065) q[7];
rz(-2.9646706038961157) q[7];