OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(0.7989646072495342) q[0];
rz(1.471288842647148) q[0];
ry(-1.4930086482657392) q[1];
rz(0.4738182242232648) q[1];
ry(1.4794289948726875) q[2];
rz(2.3246635034414607) q[2];
ry(0.9956153954914608) q[3];
rz(0.6507185014005596) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.4525416022191795) q[0];
rz(0.3144735095004801) q[0];
ry(1.7983700714665956) q[1];
rz(1.73821526908359) q[1];
ry(0.21771507979665794) q[2];
rz(-0.1082938323944056) q[2];
ry(1.5240811334098663) q[3];
rz(-0.035830580748541514) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.3401671302132447) q[0];
rz(1.291710665766438) q[0];
ry(-2.78461931968806) q[1];
rz(0.815676214203136) q[1];
ry(2.2512277593914747) q[2];
rz(-1.028470487664518) q[2];
ry(1.548456687400349) q[3];
rz(-2.6305736325321467) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.282971035280518) q[0];
rz(2.5195982279204334) q[0];
ry(0.2901698632992451) q[1];
rz(2.407292415314424) q[1];
ry(-2.452778899941683) q[2];
rz(1.8642049455770273) q[2];
ry(-1.595358437019602) q[3];
rz(2.5192167962964525) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.3988959906767747) q[0];
rz(1.034260032015679) q[0];
ry(-1.751998844058394) q[1];
rz(-2.845428198340694) q[1];
ry(0.03263964730752189) q[2];
rz(-0.36718433647662085) q[2];
ry(-0.5869458605910518) q[3];
rz(-2.53441821077511) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.8792715527952715) q[0];
rz(-0.7256418863020775) q[0];
ry(-1.7270458508772562) q[1];
rz(2.9871717134907043) q[1];
ry(-0.0052288300659393905) q[2];
rz(1.131139696182823) q[2];
ry(-1.6806899865015639) q[3];
rz(2.793576116097329) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.8083209356594416) q[0];
rz(2.5822490063569417) q[0];
ry(1.993336200990679) q[1];
rz(-2.1672888149375256) q[1];
ry(-1.5940212471314659) q[2];
rz(-0.5638066397339756) q[2];
ry(-0.9693769565981363) q[3];
rz(-0.2424301842245243) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.922635035859793) q[0];
rz(2.5406463051382575) q[0];
ry(-1.4813991455985223) q[1];
rz(-1.8968709835486168) q[1];
ry(-0.900113195467112) q[2];
rz(-0.906989230823594) q[2];
ry(1.5064334949806453) q[3];
rz(-2.456484116920134) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.6143561766356884) q[0];
rz(-1.51924805768052) q[0];
ry(-3.0652997524571703) q[1];
rz(-1.3969087733372987) q[1];
ry(-1.5646708681089887) q[2];
rz(-2.275138560042321) q[2];
ry(1.5920854911440596) q[3];
rz(-2.4133645204305254) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.638591416634328) q[0];
rz(0.1310135868934772) q[0];
ry(-0.567784521083411) q[1];
rz(-1.3814156533772621) q[1];
ry(-2.3403024554829552) q[2];
rz(0.814100579725286) q[2];
ry(0.37401155401475794) q[3];
rz(0.7264701077090661) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.07342917661927206) q[0];
rz(-0.7149703378204006) q[0];
ry(1.5941472758622026) q[1];
rz(2.575138027530702) q[1];
ry(-1.332969863663814) q[2];
rz(2.7827164113751843) q[2];
ry(2.9513148029912495) q[3];
rz(-0.2652547817365505) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.8324317771289205) q[0];
rz(1.7200748478589922) q[0];
ry(1.4059896816323123) q[1];
rz(-2.5738538995411107) q[1];
ry(-1.7965208208343764) q[2];
rz(3.012019805184429) q[2];
ry(-0.8471494807635801) q[3];
rz(-2.9943563999390346) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.7772256758808158) q[0];
rz(0.7195526235526293) q[0];
ry(-3.100670382389256) q[1];
rz(1.0756881329620693) q[1];
ry(-1.9589975370283024) q[2];
rz(1.4347648145239513) q[2];
ry(-2.6721090402938725) q[3];
rz(-1.5838325474501591) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.166657895429055) q[0];
rz(-0.39991240244870685) q[0];
ry(-0.5330711132749903) q[1];
rz(-0.9490460571761075) q[1];
ry(2.1780254360333684) q[2];
rz(2.8780157679990945) q[2];
ry(-2.751535646575156) q[3];
rz(-1.8372432463359458) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.1278515813047805) q[0];
rz(-0.08630857555006875) q[0];
ry(-2.4437389476993356) q[1];
rz(-1.2174839424994397) q[1];
ry(1.3760234832202052) q[2];
rz(-1.5100631165765623) q[2];
ry(0.5724025891391964) q[3];
rz(1.4510315841952242) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.6000199026235629) q[0];
rz(2.3191701271955973) q[0];
ry(1.2137326389504592) q[1];
rz(0.9278782841704546) q[1];
ry(2.0824142876899647) q[2];
rz(-1.2377643226695383) q[2];
ry(-2.5161528061588196) q[3];
rz(2.4741688471296186) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.3146786509516946) q[0];
rz(-0.7776309532781295) q[0];
ry(-0.1526766250562197) q[1];
rz(2.9105738275509405) q[1];
ry(1.9332800834369115) q[2];
rz(2.9597511745358704) q[2];
ry(0.6624370687361001) q[3];
rz(-0.08731831714847046) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.4953607067720549) q[0];
rz(-1.2014068226206824) q[0];
ry(2.122187381739224) q[1];
rz(1.7758997578483058) q[1];
ry(2.5007009979682397) q[2];
rz(0.3894402472001154) q[2];
ry(-2.985996657333595) q[3];
rz(-2.3447090224167133) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.7827107039313272) q[0];
rz(1.0817743798185733) q[0];
ry(0.42563040486013753) q[1];
rz(0.2790043229412207) q[1];
ry(-2.2405242564959362) q[2];
rz(0.9171031463815096) q[2];
ry(-1.6876390934860686) q[3];
rz(0.11124183430621616) q[3];