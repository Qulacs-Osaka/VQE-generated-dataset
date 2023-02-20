OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-3.078937272704774) q[0];
rz(2.512100228341042) q[0];
ry(3.090104800112652) q[1];
rz(-1.7665020768204407) q[1];
ry(-1.270065605427802) q[2];
rz(2.223238157440661) q[2];
ry(3.135959290743077) q[3];
rz(1.8179902645043722) q[3];
ry(3.1275855977011573) q[4];
rz(-2.2835716883166324) q[4];
ry(-3.139157547519748) q[5];
rz(-0.9000187707837674) q[5];
ry(1.5715409049287017) q[6];
rz(-2.9955224525628297) q[6];
ry(1.5701641455257407) q[7];
rz(-0.11025072611582669) q[7];
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
ry(1.609671928036982) q[0];
rz(-1.137544681141045) q[0];
ry(1.4807727369130568) q[1];
rz(2.3349264456424614) q[1];
ry(2.664735946596504) q[2];
rz(2.3063708344383236) q[2];
ry(-1.591240622188904) q[3];
rz(-3.1407445201952484) q[3];
ry(-1.5658016775393477) q[4];
rz(-0.6629058398529106) q[4];
ry(1.5703945494191192) q[5];
rz(1.499806264566014) q[5];
ry(-2.1676042421741393) q[6];
rz(3.049145435079622) q[6];
ry(-1.9083332546857812) q[7];
rz(-2.845243479537197) q[7];
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
ry(-1.0345834671278065) q[0];
rz(2.223826462117775) q[0];
ry(-2.1174523347955336) q[1];
rz(1.07598117444238) q[1];
ry(1.5593815379299558) q[2];
rz(-2.3286335271216587) q[2];
ry(-1.5584953017951213) q[3];
rz(0.9879744871481044) q[3];
ry(-0.006322403490165662) q[4];
rz(-1.1781171198814295) q[4];
ry(1.829733920729798) q[5];
rz(2.6200912450675933) q[5];
ry(0.04687093121250818) q[6];
rz(0.09200276675650634) q[6];
ry(0.06014672502677332) q[7];
rz(2.9110452670615605) q[7];
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
ry(0.23423784016137486) q[0];
rz(-2.9783547979949345) q[0];
ry(-1.4472780638341725) q[1];
rz(3.127365489790087) q[1];
ry(-3.1323986027096007) q[2];
rz(-0.7559807418428547) q[2];
ry(3.137724813921042) q[3];
rz(1.5566072126501072) q[3];
ry(0.0006353257942297574) q[4];
rz(-1.055136198596803) q[4];
ry(3.134121712953932) q[5];
rz(-0.5399191405944697) q[5];
ry(-0.529892629178891) q[6];
rz(2.902639410841188) q[6];
ry(-2.7381420487970525) q[7];
rz(-2.7024390223142642) q[7];
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
ry(0.13367805231796734) q[0];
rz(-0.7417519800905399) q[0];
ry(-2.4873046914782133) q[1];
rz(2.136373356261249) q[1];
ry(1.566963705018553) q[2];
rz(0.11480792334613898) q[2];
ry(-2.221956787481097) q[3];
rz(-2.74764874124763) q[3];
ry(-3.1265113623282805) q[4];
rz(1.0021155249606375) q[4];
ry(-2.265406527461161) q[5];
rz(3.0881574579246234) q[5];
ry(-3.1231890699854263) q[6];
rz(3.101807593965451) q[6];
ry(-0.5275214920615802) q[7];
rz(-1.514705550958761) q[7];
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
ry(2.0408485131944376) q[0];
rz(0.9439948516001259) q[0];
ry(-2.9827015835518336) q[1];
rz(-0.6748989367803143) q[1];
ry(-1.5787287849560314) q[2];
rz(-0.8750737139186171) q[2];
ry(-1.547998217900167) q[3];
rz(1.7415515106779944) q[3];
ry(0.001413250957033263) q[4];
rz(0.9516851299325317) q[4];
ry(3.100609189985887) q[5];
rz(-1.4524923208003366) q[5];
ry(1.4480482340911585) q[6];
rz(1.7635250832891916) q[6];
ry(1.3799552404097264) q[7];
rz(1.1585781377258078) q[7];
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
ry(-0.9666229099414156) q[0];
rz(-2.943098122777924) q[0];
ry(0.7121427326113466) q[1];
rz(-1.8964804919565195) q[1];
ry(-0.006648022101132689) q[2];
rz(2.7214426477975087) q[2];
ry(-0.0032485260371613884) q[3];
rz(-2.4690047021983896) q[3];
ry(-3.1029966927732513) q[4];
rz(0.1362193118421171) q[4];
ry(-3.1393234013373013) q[5];
rz(-3.055006097645361) q[5];
ry(0.5781819339936067) q[6];
rz(-0.10560785723601514) q[6];
ry(1.942029864432738) q[7];
rz(2.648242015266088) q[7];
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
ry(1.1062655564623665) q[0];
rz(2.957316328839533) q[0];
ry(-2.3294983236637754) q[1];
rz(-2.043093374846186) q[1];
ry(0.005411547598844102) q[2];
rz(1.3057730391465567) q[2];
ry(-0.09639646196457628) q[3];
rz(-0.8496667181594608) q[3];
ry(1.543715674258322) q[4];
rz(-3.140905901443089) q[4];
ry(1.5709257389373965) q[5];
rz(3.0979859366845246) q[5];
ry(-1.5410607068398665) q[6];
rz(-3.077750468969374) q[6];
ry(0.28238799127908576) q[7];
rz(-2.580501084464335) q[7];
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
ry(2.033748663848079) q[0];
rz(0.5852674877515079) q[0];
ry(2.7721519022853585) q[1];
rz(-0.16232622094792062) q[1];
ry(-1.5408027581625552) q[2];
rz(-0.006707172105913652) q[2];
ry(1.5760283705033542) q[3];
rz(1.7372844375415204) q[3];
ry(-1.5776632333318252) q[4];
rz(1.5542829206281503) q[4];
ry(1.5678175770915574) q[5];
rz(2.920669360817099) q[5];
ry(1.5721097287601344) q[6];
rz(0.02588496180663494) q[6];
ry(-1.6349361740837765) q[7];
rz(-1.570156073363144) q[7];
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
ry(-0.13299053638386119) q[0];
rz(0.7132825894735236) q[0];
ry(-0.011790574322775757) q[1];
rz(-2.9250433796616893) q[1];
ry(-1.57107488906569) q[2];
rz(3.1411695423400285) q[2];
ry(0.0017760688024862576) q[3];
rz(1.401783233198115) q[3];
ry(-3.0737552724857498) q[4];
rz(-1.3177195306893221) q[4];
ry(0.006966774856633342) q[5];
rz(0.21805943172809883) q[5];
ry(-1.6215085404817788) q[6];
rz(-0.002134913583517523) q[6];
ry(1.5706444805454456) q[7];
rz(-1.9226426018020486) q[7];
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
ry(-1.587953710883542) q[0];
rz(0.14585876318393645) q[0];
ry(0.0007255411380800998) q[1];
rz(-2.4735134203996805) q[1];
ry(1.5407787100402495) q[2];
rz(1.0897852306130023) q[2];
ry(3.038438299875067) q[3];
rz(-3.0444279416134092) q[3];
ry(3.141553424590285) q[4];
rz(0.27625685000871714) q[4];
ry(-0.45690240518056996) q[5];
rz(3.1323056946378496) q[5];
ry(-1.5764426326151035) q[6];
rz(2.922361169620842) q[6];
ry(3.134345372228308) q[7];
rz(2.731701038480942) q[7];
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
ry(3.137596103596536) q[0];
rz(-1.4422267427136193) q[0];
ry(0.0998235494105573) q[1];
rz(1.2492330270251915) q[1];
ry(0.00038314202949298303) q[2];
rz(-1.0943024790453642) q[2];
ry(0.0007387745684903634) q[3];
rz(-1.4485539972386645) q[3];
ry(1.5698460536863925) q[4];
rz(-2.1348637961293053) q[4];
ry(-0.01795850144738953) q[5];
rz(1.6151691186143804) q[5];
ry(-1.5527778028029005) q[6];
rz(-1.5599379669097269) q[6];
ry(-1.5729155273732873) q[7];
rz(-1.571491638616192) q[7];
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
ry(-0.9342259153193329) q[0];
rz(-3.0867339635992117) q[0];
ry(-0.0006991194398525181) q[1];
rz(2.1131865014423434) q[1];
ry(0.07776896102982978) q[2];
rz(1.5765631148683263) q[2];
ry(-3.1411500400709107) q[3];
rz(0.22231689637667973) q[3];
ry(-3.140490370303851) q[4];
rz(0.7872110355243898) q[4];
ry(-0.07972949073768199) q[5];
rz(1.5352943815439186) q[5];
ry(-1.577098682002144) q[6];
rz(1.573977795344044) q[6];
ry(-1.564328757367873) q[7];
rz(1.624723701660538) q[7];
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
ry(-3.076797992831344) q[0];
rz(-3.111039432226767) q[0];
ry(1.6205635477903169) q[1];
rz(-0.07847041318776751) q[1];
ry(1.5707419238669946) q[2];
rz(0.9847999812225927) q[2];
ry(1.5712606812160415) q[3];
rz(-0.0818229240744257) q[3];
ry(3.141067272550507) q[4];
rz(1.6097960998236767) q[4];
ry(1.5703453800015987) q[5];
rz(-1.6999219487318649) q[5];
ry(-0.9152603969533863) q[6];
rz(3.1375975784602463) q[6];
ry(-0.01717246720190724) q[7];
rz(2.70991338575048) q[7];
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
ry(1.5723827382069482) q[0];
rz(1.5332533806110975) q[0];
ry(-1.6116888867859194) q[1];
rz(-0.7864799004346601) q[1];
ry(-3.141426434753099) q[2];
rz(2.638472325091817) q[2];
ry(3.11623054484946) q[3];
rz(3.062529383158614) q[3];
ry(0.0002747016740282424) q[4];
rz(2.8871822294871663) q[4];
ry(3.141422950330438) q[5];
rz(-1.7029080642468175) q[5];
ry(1.5702576066633878) q[6];
rz(-3.139237370920684) q[6];
ry(3.141299307406045) q[7];
rz(-0.3742798426289689) q[7];
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
ry(-1.4802983840167405) q[0];
rz(0.3385919514388958) q[0];
ry(-0.0009129111941644054) q[1];
rz(0.23622026055907064) q[1];
ry(-0.0011073688877916155) q[2];
rz(1.8240114812969095) q[2];
ry(1.5694942077957057) q[3];
rz(-2.1067106193399923) q[3];
ry(1.5707610929752984) q[4];
rz(-1.2299166320464048) q[4];
ry(1.56216490004096) q[5];
rz(2.594808911517757) q[5];
ry(-2.2247096073791) q[6];
rz(1.9124478106659044) q[6];
ry(1.570796116127016) q[7];
rz(-0.6390402848072076) q[7];