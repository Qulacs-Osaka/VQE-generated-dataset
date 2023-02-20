OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-2.742663389357872) q[0];
rz(-2.148063551096595) q[0];
ry(0.3440536594947754) q[1];
rz(1.5674923289100855) q[1];
ry(1.7791068426038623) q[2];
rz(-2.5847684852101924) q[2];
ry(1.255059223806737) q[3];
rz(1.4999711936776734) q[3];
ry(2.022084793249605) q[4];
rz(-0.02135956846443232) q[4];
ry(-2.4174178304945624) q[5];
rz(-0.8269524783289904) q[5];
ry(1.2672726307861772) q[6];
rz(0.9759033922215012) q[6];
ry(2.069779336591843) q[7];
rz(-0.1414391560497192) q[7];
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
ry(0.8633024630345469) q[0];
rz(2.4881406236722414) q[0];
ry(0.9721323778842511) q[1];
rz(0.01924534596426852) q[1];
ry(0.8851522777368945) q[2];
rz(2.2983977784930723) q[2];
ry(-1.4569240023345111) q[3];
rz(3.0278768248311834) q[3];
ry(-2.7542317960939973) q[4];
rz(-0.7464676673936984) q[4];
ry(-1.5528223161100598) q[5];
rz(-0.0017649560031019397) q[5];
ry(-1.9666264330834835) q[6];
rz(-1.7617510572830462) q[6];
ry(-2.9625811572126683) q[7];
rz(0.5672790259267559) q[7];
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
ry(-0.5803735877994987) q[0];
rz(0.0377625209811374) q[0];
ry(-1.5312399957641658) q[1];
rz(-2.61417896988653) q[1];
ry(-0.4409013970967442) q[2];
rz(-2.226446722222036) q[2];
ry(0.8968325321199504) q[3];
rz(-0.5073615341528332) q[3];
ry(2.637389932652251) q[4];
rz(-0.8327302434305988) q[4];
ry(-1.6636692693390531) q[5];
rz(-2.231876555741729) q[5];
ry(-2.1280138824329917) q[6];
rz(-0.9549963428584453) q[6];
ry(-0.6519513005222458) q[7];
rz(2.5285801000905206) q[7];
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
ry(-1.4588435923625551) q[0];
rz(1.5508788040035082) q[0];
ry(2.575959355505816) q[1];
rz(-1.6184378766841272) q[1];
ry(2.3570118372557576) q[2];
rz(0.08242310826435927) q[2];
ry(2.1784103567414475) q[3];
rz(0.8242820777333081) q[3];
ry(-2.201240215229027) q[4];
rz(1.2064364458654282) q[4];
ry(-0.7390725939493628) q[5];
rz(-0.9955332371621444) q[5];
ry(-0.9603572727435681) q[6];
rz(-1.3760361855123262) q[6];
ry(2.8076266750078047) q[7];
rz(2.2479398273472753) q[7];
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
ry(-0.6481582155673395) q[0];
rz(2.6825336164113334) q[0];
ry(0.707458093143716) q[1];
rz(1.5677134577083018) q[1];
ry(-1.5879813985173126) q[2];
rz(0.9877854068705207) q[2];
ry(-1.1128837608863076) q[3];
rz(1.6483633647691016) q[3];
ry(0.9176742888327816) q[4];
rz(-1.2805333949970121) q[4];
ry(0.9951757350866344) q[5];
rz(-2.5825212927223475) q[5];
ry(-2.223921181383694) q[6];
rz(3.0886567074152276) q[6];
ry(-1.983230319125754) q[7];
rz(0.5864589250598623) q[7];
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
ry(0.5984873837741578) q[0];
rz(-1.4360243059542857) q[0];
ry(-0.513117229730302) q[1];
rz(-0.3169403153133235) q[1];
ry(-2.1039116571001606) q[2];
rz(-1.5886111962690077) q[2];
ry(1.534787348133106) q[3];
rz(0.5154500139909802) q[3];
ry(-1.1592135953128135) q[4];
rz(-1.1739904904773424) q[4];
ry(2.837732273225854) q[5];
rz(-1.9896333116406262) q[5];
ry(-1.0209710156023126) q[6];
rz(-1.9497973523267957) q[6];
ry(-1.0761053065969792) q[7];
rz(-2.8117710942793788) q[7];
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
ry(-0.858583654448609) q[0];
rz(-1.7238896445490852) q[0];
ry(0.2864694464931446) q[1];
rz(1.554404387178147) q[1];
ry(1.8778768385317877) q[2];
rz(-0.7673450667931636) q[2];
ry(1.2678206335215316) q[3];
rz(-2.9722100438052323) q[3];
ry(-3.040324899150136) q[4];
rz(-0.4682970450174898) q[4];
ry(-2.139007909573169) q[5];
rz(-0.918037389494147) q[5];
ry(0.338403216369492) q[6];
rz(-0.7685198947990172) q[6];
ry(-0.7635717253618842) q[7];
rz(0.8565926344164829) q[7];
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
ry(-0.5037973312535902) q[0];
rz(-1.1293883838198329) q[0];
ry(-0.11543996842730049) q[1];
rz(0.7726084866808921) q[1];
ry(-2.0794860909721917) q[2];
rz(-2.596362861177375) q[2];
ry(2.1327416284287946) q[3];
rz(-2.3623818037328315) q[3];
ry(-1.952771204526221) q[4];
rz(3.0787464115712995) q[4];
ry(-2.5741600817741737) q[5];
rz(-2.2248306554143666) q[5];
ry(0.9759752523305966) q[6];
rz(3.1170047670904166) q[6];
ry(-2.603645054892349) q[7];
rz(-2.23863310031735) q[7];
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
ry(0.5081912543563822) q[0];
rz(0.9287460637780832) q[0];
ry(-2.490102858994984) q[1];
rz(-1.0136424229742218) q[1];
ry(-2.8576135209753937) q[2];
rz(-0.7219645386249843) q[2];
ry(-1.6977086539738087) q[3];
rz(-1.7774715250844582) q[3];
ry(2.941136194573824) q[4];
rz(-3.085402689882036) q[4];
ry(-1.3877106814820213) q[5];
rz(-2.395073153185594) q[5];
ry(1.555778976792782) q[6];
rz(-0.5404298995681004) q[6];
ry(1.4720585141831508) q[7];
rz(-3.052248982751877) q[7];
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
ry(-2.0809142061967383) q[0];
rz(-2.069788335877327) q[0];
ry(2.989529084389686) q[1];
rz(2.232441656548776) q[1];
ry(-0.840469903960881) q[2];
rz(-0.6975048972785388) q[2];
ry(-2.6005156444292505) q[3];
rz(-2.9796920979134183) q[3];
ry(1.0938995588106677) q[4];
rz(-0.0022481548778708325) q[4];
ry(-1.791450945477341) q[5];
rz(2.8081406971003595) q[5];
ry(1.0598614248414897) q[6];
rz(-2.7158649472071508) q[6];
ry(2.299967175233788) q[7];
rz(-2.439492964296995) q[7];
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
ry(-2.2335136690588273) q[0];
rz(-2.919443991624837) q[0];
ry(-0.249198463642756) q[1];
rz(1.5691406818231655) q[1];
ry(-1.8394208414618425) q[2];
rz(0.280435008344031) q[2];
ry(-0.5708995426493386) q[3];
rz(1.1401752448211966) q[3];
ry(2.7318277037700547) q[4];
rz(-1.2475578412268695) q[4];
ry(-1.7289637985730941) q[5];
rz(-2.221306025690054) q[5];
ry(0.41866398499503443) q[6];
rz(1.537042340885045) q[6];
ry(-1.6651722127198) q[7];
rz(0.03642032458487635) q[7];
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
ry(2.6015004664572805) q[0];
rz(1.9174119385904844) q[0];
ry(-0.6638458333961826) q[1];
rz(0.8713213371261325) q[1];
ry(-1.8987183489457848) q[2];
rz(2.143175137261246) q[2];
ry(1.7043978663676524) q[3];
rz(-0.36239786145880026) q[3];
ry(0.740290463914663) q[4];
rz(-1.202999793576975) q[4];
ry(-2.756216764641661) q[5];
rz(-0.08788092687534464) q[5];
ry(-2.0221452443439123) q[6];
rz(-1.6991450091789628) q[6];
ry(2.7589878562084955) q[7];
rz(-1.7653560942120259) q[7];
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
ry(-2.236842976276927) q[0];
rz(0.14716663096479898) q[0];
ry(2.5431319761197364) q[1];
rz(2.158212080872552) q[1];
ry(2.982807130169247) q[2];
rz(0.325148675214141) q[2];
ry(-1.8758538833382572) q[3];
rz(2.4460632117227346) q[3];
ry(-1.3929360949066147) q[4];
rz(2.3903954194034798) q[4];
ry(2.7708820996040426) q[5];
rz(0.9111841907427854) q[5];
ry(-1.3636471066199927) q[6];
rz(2.664568053209765) q[6];
ry(-1.8994339145147483) q[7];
rz(-0.8361966329534258) q[7];
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
ry(0.4683810559113022) q[0];
rz(0.41091784229898476) q[0];
ry(-0.6446251453439225) q[1];
rz(0.09436944047526089) q[1];
ry(1.920389714567475) q[2];
rz(1.9913400106522792) q[2];
ry(-1.7192217930762408) q[3];
rz(1.909022299315527) q[3];
ry(1.2905857818009512) q[4];
rz(2.3246909597350243) q[4];
ry(0.5090221962437346) q[5];
rz(-2.5481376796945368) q[5];
ry(-1.4540962552706242) q[6];
rz(-1.744759963671041) q[6];
ry(-3.0650606923275596) q[7];
rz(2.2675730103804574) q[7];
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
ry(1.7115120444353473) q[0];
rz(1.2204301545871377) q[0];
ry(0.7780415835714277) q[1];
rz(3.0427137303363985) q[1];
ry(2.1377616814090086) q[2];
rz(2.7357232857470875) q[2];
ry(0.29995198260211586) q[3];
rz(0.3437032406797283) q[3];
ry(-0.40303921013501626) q[4];
rz(-2.532245712577258) q[4];
ry(-2.460967565524043) q[5];
rz(1.0046921688642443) q[5];
ry(-1.4997407494372084) q[6];
rz(-2.8786848963634357) q[6];
ry(-3.0697035047045667) q[7];
rz(-0.019274142994269754) q[7];
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
ry(2.0641489750041906) q[0];
rz(-0.8058683104910623) q[0];
ry(0.7324543432393431) q[1];
rz(0.1414862947087405) q[1];
ry(0.6217481533277254) q[2];
rz(1.632629071617783) q[2];
ry(-1.8305600828817017) q[3];
rz(2.68313096583076) q[3];
ry(-0.6983613966028782) q[4];
rz(2.4721400632922457) q[4];
ry(-0.8122311570840797) q[5];
rz(2.295044570712027) q[5];
ry(1.7275137402123033) q[6];
rz(2.0128872965924502) q[6];
ry(-2.3473562783033564) q[7];
rz(3.022051772642145) q[7];
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
ry(-2.8180259931265925) q[0];
rz(-0.02751801670277698) q[0];
ry(-2.5085401958966678) q[1];
rz(-0.3595906470168196) q[1];
ry(-2.6480145844594905) q[2];
rz(-1.3514509968445518) q[2];
ry(-1.314441873698175) q[3];
rz(1.6114166601991426) q[3];
ry(1.9800097083618269) q[4];
rz(-1.7244417757465833) q[4];
ry(0.26358594780357336) q[5];
rz(-2.705371412249751) q[5];
ry(-0.5357391713732098) q[6];
rz(2.102987666751039) q[6];
ry(-1.0688199052042038) q[7];
rz(1.5128539099711835) q[7];
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
ry(-0.37822894653595807) q[0];
rz(1.4943491578965922) q[0];
ry(2.314525498277353) q[1];
rz(-3.047481011263594) q[1];
ry(-2.115750059937895) q[2];
rz(2.0950175334353363) q[2];
ry(-2.2756279266517314) q[3];
rz(2.318276786034255) q[3];
ry(0.695343762320233) q[4];
rz(1.800901744045201) q[4];
ry(2.91486902573545) q[5];
rz(-2.168187916005989) q[5];
ry(2.238370429446106) q[6];
rz(-1.2534683137579592) q[6];
ry(-1.1452805964233823) q[7];
rz(2.6755890723837052) q[7];
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
ry(0.7338826288118643) q[0];
rz(2.9170734313985327) q[0];
ry(1.1664085844264633) q[1];
rz(-0.033386882809734834) q[1];
ry(-0.5243325623262063) q[2];
rz(-1.440541424710502) q[2];
ry(-1.3936553635636473) q[3];
rz(-2.2982189697584174) q[3];
ry(-1.8413217535064854) q[4];
rz(-0.6374367401034764) q[4];
ry(-2.197026251735537) q[5];
rz(-0.23517984644794773) q[5];
ry(-1.2387105795371758) q[6];
rz(0.424471424451343) q[6];
ry(1.0214424247626255) q[7];
rz(3.0570249957276734) q[7];
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
ry(1.5288626622693005) q[0];
rz(-2.2694238943661835) q[0];
ry(-2.6114848128025097) q[1];
rz(0.6789277772989744) q[1];
ry(-2.3368431788671864) q[2];
rz(-0.37021955536895573) q[2];
ry(-2.7817095097542524) q[3];
rz(-3.0913307376198307) q[3];
ry(-1.8156926693021742) q[4];
rz(-0.653822394584832) q[4];
ry(-2.6289395905730806) q[5];
rz(-1.8201011292324116) q[5];
ry(-1.6782371725423053) q[6];
rz(-3.0362441286336637) q[6];
ry(1.074087486507456) q[7];
rz(1.5405215600899984) q[7];
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
ry(-2.4844077901633836) q[0];
rz(-2.7919395074039515) q[0];
ry(-0.33263358338286353) q[1];
rz(-0.6366668469164294) q[1];
ry(1.797797172345235) q[2];
rz(-1.8610342686409629) q[2];
ry(-1.8495387836726607) q[3];
rz(-0.12173890392960575) q[3];
ry(-3.032909148252383) q[4];
rz(1.1984961830661602) q[4];
ry(-2.8587142847867546) q[5];
rz(-2.804457205960298) q[5];
ry(-0.46743354265135245) q[6];
rz(-1.799956733222258) q[6];
ry(2.9105121866014634) q[7];
rz(-0.8756548069179955) q[7];
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
ry(2.3693853578573987) q[0];
rz(1.7325378277074694) q[0];
ry(2.279185085080681) q[1];
rz(-2.521380142848152) q[1];
ry(-1.99496538395731) q[2];
rz(0.5957331312928522) q[2];
ry(1.7875495894941722) q[3];
rz(1.476036831474702) q[3];
ry(2.2852235104381133) q[4];
rz(-2.7058288956815666) q[4];
ry(1.7400286968539396) q[5];
rz(-1.4947652843271797) q[5];
ry(0.8735920850355958) q[6];
rz(-2.912476362110812) q[6];
ry(-0.4374001467626041) q[7];
rz(2.8133492528953594) q[7];
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
ry(0.8023661740880635) q[0];
rz(1.481341795242832) q[0];
ry(-0.30167048022808185) q[1];
rz(-2.6248589672097) q[1];
ry(-2.701052714298043) q[2];
rz(0.2373107223368205) q[2];
ry(-1.2816934223089707) q[3];
rz(1.7791455060126182) q[3];
ry(1.3769207854105119) q[4];
rz(0.6278767343745216) q[4];
ry(-0.26061554363005696) q[5];
rz(2.459831700671389) q[5];
ry(-1.9471444051301163) q[6];
rz(-2.5557724301139646) q[6];
ry(-0.7133503112746427) q[7];
rz(-0.24824685389570927) q[7];
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
ry(-2.4861138927501916) q[0];
rz(0.504509020374285) q[0];
ry(-1.2953899408158325) q[1];
rz(-2.827730268732054) q[1];
ry(1.593221575476346) q[2];
rz(-0.22058369777564127) q[2];
ry(1.0960128302500942) q[3];
rz(1.4001836815935178) q[3];
ry(0.2642634499370068) q[4];
rz(1.4652154088484535) q[4];
ry(-1.2975841722146353) q[5];
rz(2.631175904049815) q[5];
ry(-1.6966744780542873) q[6];
rz(1.022749320359681) q[6];
ry(0.5373783105230111) q[7];
rz(-0.5640909742741149) q[7];
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
ry(-2.821526667437314) q[0];
rz(0.5052846654502243) q[0];
ry(0.6715758153575655) q[1];
rz(-2.5041110550715477) q[1];
ry(2.2975278675433226) q[2];
rz(1.908375318707221) q[2];
ry(2.2186834851109927) q[3];
rz(-1.1366952807699935) q[3];
ry(-1.2988744667962075) q[4];
rz(-1.587342555191813) q[4];
ry(2.2087442755683115) q[5];
rz(0.2653185346719624) q[5];
ry(0.23823049619496284) q[6];
rz(-2.4135367908897067) q[6];
ry(-1.339088116846553) q[7];
rz(-0.4332424092261187) q[7];
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
ry(1.5932718574970257) q[0];
rz(1.621788801911503) q[0];
ry(-0.30003031343462006) q[1];
rz(-2.023560880794581) q[1];
ry(-2.300801940021387) q[2];
rz(2.86937237417333) q[2];
ry(-0.4133995251324656) q[3];
rz(2.486042324002541) q[3];
ry(-2.2423893514805044) q[4];
rz(1.0301218237654355) q[4];
ry(-0.4120675043867336) q[5];
rz(0.7078156065211136) q[5];
ry(0.8615150119851246) q[6];
rz(-1.8397520776455405) q[6];
ry(-1.2896953539456826) q[7];
rz(-0.8175330590761087) q[7];
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
ry(0.5300274655734412) q[0];
rz(-1.3196420188798967) q[0];
ry(-1.164732005042695) q[1];
rz(2.3078670578126923) q[1];
ry(1.9803246539136463) q[2];
rz(1.6524202075471752) q[2];
ry(-1.2210514167179718) q[3];
rz(-1.9020674576678678) q[3];
ry(1.0061958058940617) q[4];
rz(-3.0754742406741036) q[4];
ry(-1.0545662015415465) q[5];
rz(-2.0943874573186547) q[5];
ry(2.277672124400307) q[6];
rz(2.6696627066422143) q[6];
ry(-0.6273874274594627) q[7];
rz(0.08584469470125011) q[7];
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
ry(-1.2910343809335316) q[0];
rz(0.64068033631803) q[0];
ry(-2.2678735060172697) q[1];
rz(-2.377132135497277) q[1];
ry(-1.8670072286695747) q[2];
rz(0.9260259055170637) q[2];
ry(2.8786191817334186) q[3];
rz(-0.00893726087454727) q[3];
ry(2.666737807743747) q[4];
rz(-2.387613279713768) q[4];
ry(0.17846919628156233) q[5];
rz(2.2232678964124215) q[5];
ry(-1.180851532516677) q[6];
rz(2.4035434381872824) q[6];
ry(-0.5560525600296348) q[7];
rz(2.3620517646284815) q[7];