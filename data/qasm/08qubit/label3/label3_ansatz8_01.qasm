OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(0.3782074055555693) q[0];
ry(-2.018809981314919) q[1];
cx q[0],q[1];
ry(-2.358729078257783) q[0];
ry(-1.8620115066893916) q[1];
cx q[0],q[1];
ry(-0.6455935110831899) q[2];
ry(-2.9416549925133832) q[3];
cx q[2],q[3];
ry(-0.36166983923515167) q[2];
ry(-0.6007015564979675) q[3];
cx q[2],q[3];
ry(-1.616134303320254) q[4];
ry(-2.111246996307188) q[5];
cx q[4],q[5];
ry(-1.0515440151413962) q[4];
ry(2.5455721809157814) q[5];
cx q[4],q[5];
ry(0.04776970105060003) q[6];
ry(-2.1795822097963153) q[7];
cx q[6],q[7];
ry(-2.058297744907417) q[6];
ry(-2.1511268502176444) q[7];
cx q[6],q[7];
ry(-0.34299796511962755) q[0];
ry(-2.478049343827176) q[2];
cx q[0],q[2];
ry(1.5678754967020894) q[0];
ry(-1.5678624601319062) q[2];
cx q[0],q[2];
ry(-2.2970634036199056) q[2];
ry(-2.2884247486484606) q[4];
cx q[2],q[4];
ry(2.217513049647372) q[2];
ry(0.5977059950943221) q[4];
cx q[2],q[4];
ry(-1.2028788721843935) q[4];
ry(-0.47652572823529304) q[6];
cx q[4],q[6];
ry(-1.7288183915252155) q[4];
ry(-1.8156024627893095) q[6];
cx q[4],q[6];
ry(-0.18571232495375123) q[1];
ry(0.28380925931631573) q[3];
cx q[1],q[3];
ry(-1.5728951518130891) q[1];
ry(1.5738057754097297) q[3];
cx q[1],q[3];
ry(-1.0734617874830013) q[3];
ry(-0.4441538024459337) q[5];
cx q[3],q[5];
ry(-2.2100516601706257) q[3];
ry(-1.3397638518003891) q[5];
cx q[3],q[5];
ry(0.08293842977238697) q[5];
ry(2.2053880520066342) q[7];
cx q[5],q[7];
ry(-2.6486222452104715) q[5];
ry(2.611599367458653) q[7];
cx q[5],q[7];
ry(0.8664996055226933) q[0];
ry(-3.097828051467514) q[1];
cx q[0],q[1];
ry(-0.15250358421445812) q[0];
ry(-0.02065344523685475) q[1];
cx q[0],q[1];
ry(2.629153542585973) q[2];
ry(0.8602905898397565) q[3];
cx q[2],q[3];
ry(-0.4732443562827065) q[2];
ry(-3.0535084123737213) q[3];
cx q[2],q[3];
ry(-0.2531203018992876) q[4];
ry(-2.12950943084242) q[5];
cx q[4],q[5];
ry(0.5804709503022507) q[4];
ry(1.9942131100713079) q[5];
cx q[4],q[5];
ry(2.7588065567857245) q[6];
ry(-1.2281423205827995) q[7];
cx q[6],q[7];
ry(0.513791004187678) q[6];
ry(0.9818942441770202) q[7];
cx q[6],q[7];
ry(-3.107325001701549) q[0];
ry(0.5532220355581892) q[2];
cx q[0],q[2];
ry(0.0005463138559127501) q[0];
ry(3.1410731447762887) q[2];
cx q[0],q[2];
ry(1.7680212728013096) q[2];
ry(0.7434353225790389) q[4];
cx q[2],q[4];
ry(0.17586726084462628) q[2];
ry(0.3582138742485205) q[4];
cx q[2],q[4];
ry(1.7625516237123526) q[4];
ry(-2.5488415295634237) q[6];
cx q[4],q[6];
ry(-0.45965738764803615) q[4];
ry(-0.3434109987187218) q[6];
cx q[4],q[6];
ry(-2.8729329817827667) q[1];
ry(3.016920511656075) q[3];
cx q[1],q[3];
ry(-1.5644634604226164) q[1];
ry(-0.6982741372507046) q[3];
cx q[1],q[3];
ry(-1.1204537901446416) q[3];
ry(-2.5214361855280636) q[5];
cx q[3],q[5];
ry(3.1364558579255997) q[3];
ry(-3.113183503506714) q[5];
cx q[3],q[5];
ry(1.9642311463988422) q[5];
ry(1.731313416948768) q[7];
cx q[5],q[7];
ry(0.6809936099267331) q[5];
ry(-0.16182752103683062) q[7];
cx q[5],q[7];
ry(-0.5784180886507191) q[0];
ry(1.3749430994790641) q[1];
cx q[0],q[1];
ry(0.005355938022985285) q[0];
ry(0.007551951510718847) q[1];
cx q[0],q[1];
ry(2.730206363041911) q[2];
ry(2.9045469416404335) q[3];
cx q[2],q[3];
ry(3.1373482144765457) q[2];
ry(-1.5745618868509659) q[3];
cx q[2],q[3];
ry(2.7278822851428886) q[4];
ry(-2.0648905963250987) q[5];
cx q[4],q[5];
ry(0.622034393687282) q[4];
ry(-1.0244951030206906) q[5];
cx q[4],q[5];
ry(2.393797207862045) q[6];
ry(2.8222368904748643) q[7];
cx q[6],q[7];
ry(-0.8519623959631204) q[6];
ry(-2.0098703292848374) q[7];
cx q[6],q[7];
ry(-3.1296446247300493) q[0];
ry(0.1851668347324127) q[2];
cx q[0],q[2];
ry(-1.5189157837017524) q[0];
ry(0.19796245386568953) q[2];
cx q[0],q[2];
ry(-0.017543255830550653) q[2];
ry(2.5730691086061097) q[4];
cx q[2],q[4];
ry(3.1411319360122243) q[2];
ry(-1.8072610772357226e-05) q[4];
cx q[2],q[4];
ry(0.005731782648173327) q[4];
ry(0.6804154329989197) q[6];
cx q[4],q[6];
ry(-0.7683051725052229) q[4];
ry(-0.4046444352333687) q[6];
cx q[4],q[6];
ry(1.68198099815345) q[1];
ry(-2.502805924993359) q[3];
cx q[1],q[3];
ry(0.00014573147573049283) q[1];
ry(-0.6792875414807071) q[3];
cx q[1],q[3];
ry(0.24073340193960657) q[3];
ry(-1.0668850263103875) q[5];
cx q[3],q[5];
ry(1.5714234627306967) q[3];
ry(1.922956231157082) q[5];
cx q[3],q[5];
ry(0.003305863982944678) q[5];
ry(-0.22703518717652368) q[7];
cx q[5],q[7];
ry(1.570981674457112) q[5];
ry(-1.5727272114452286) q[7];
cx q[5],q[7];
ry(-1.514145829208945) q[0];
ry(1.2922616022409636) q[1];
cx q[0],q[1];
ry(-1.5654957217574847) q[0];
ry(-3.0776852208904053) q[1];
cx q[0],q[1];
ry(-0.20182324740818558) q[2];
ry(1.1386242946885892) q[3];
cx q[2],q[3];
ry(-0.0009107064673954516) q[2];
ry(0.0005634731816943628) q[3];
cx q[2],q[3];
ry(-0.9118480578863322) q[4];
ry(-1.814499702955868) q[5];
cx q[4],q[5];
ry(1.9884790264115306) q[4];
ry(1.8124217526453064) q[5];
cx q[4],q[5];
ry(0.2659585967062916) q[6];
ry(1.5743334362713044) q[7];
cx q[6],q[7];
ry(1.348856499940744) q[6];
ry(1.5716481541925829) q[7];
cx q[6],q[7];
ry(-0.6260118693434176) q[0];
ry(-2.7982170675345066) q[2];
cx q[0],q[2];
ry(-3.0806790524925174) q[0];
ry(-3.141011039693307) q[2];
cx q[0],q[2];
ry(-0.9197027390584313) q[2];
ry(-1.9848420341492048) q[4];
cx q[2],q[4];
ry(1.572363984768475) q[2];
ry(3.139152024581118) q[4];
cx q[2],q[4];
ry(0.044217076688598485) q[4];
ry(-0.0003499271026727868) q[6];
cx q[4],q[6];
ry(1.5774560586690756) q[4];
ry(-0.00410809981765059) q[6];
cx q[4],q[6];
ry(-1.5673062465201868) q[1];
ry(-1.1294051918913048) q[3];
cx q[1],q[3];
ry(-1.5754642871841318) q[1];
ry(1.5616250098813902) q[3];
cx q[1],q[3];
ry(-0.031173085893699937) q[3];
ry(-2.2025263880480654) q[5];
cx q[3],q[5];
ry(-1.5701881022774862) q[3];
ry(3.1406912956734425) q[5];
cx q[3],q[5];
ry(-2.6998173726695627) q[5];
ry(3.138945392562716) q[7];
cx q[5],q[7];
ry(-1.566788115436758) q[5];
ry(-0.003050296200505281) q[7];
cx q[5],q[7];
ry(-2.201511987769055) q[0];
ry(1.5740453793584188) q[1];
ry(-1.0403203923024265) q[2];
ry(3.1141562412592267) q[3];
ry(3.1013998805235055) q[4];
ry(-0.4382167761993845) q[5];
ry(-1.5674170095707378) q[6];
ry(1.5740566586251719) q[7];