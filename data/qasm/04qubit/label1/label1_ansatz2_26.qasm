OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.1364969091997414) q[0];
rz(-0.23327692166053665) q[0];
ry(-1.2113232807197314) q[1];
rz(-2.310026973379529) q[1];
ry(-2.4312730578922177) q[2];
rz(-2.6846707777962773) q[2];
ry(-2.536103088732861) q[3];
rz(0.8184097015334488) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.932092395612603) q[0];
rz(2.250662869550919) q[0];
ry(-2.6090280130759935) q[1];
rz(2.4587983208333144) q[1];
ry(-2.953374619272325) q[2];
rz(-0.34612804164561517) q[2];
ry(3.0099682689199367) q[3];
rz(0.27630964463130425) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.2839823497415824) q[0];
rz(0.08716113422745853) q[0];
ry(0.3971882872100094) q[1];
rz(-2.231069820283815) q[1];
ry(-1.130973965993034) q[2];
rz(-1.0238143059731035) q[2];
ry(-0.9294493289756468) q[3];
rz(-1.063245244458499) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.480940868044401) q[0];
rz(-1.3036418596534325) q[0];
ry(0.42303157744323716) q[1];
rz(1.5335345573622003) q[1];
ry(-0.5584929336125753) q[2];
rz(2.2522564765922892) q[2];
ry(1.9915045830888547) q[3];
rz(-2.4635990615971286) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.5683501659622242) q[0];
rz(-1.0757491306886635) q[0];
ry(-0.5959681921740542) q[1];
rz(0.1618538866528736) q[1];
ry(1.649341871544572) q[2];
rz(1.3417052191323533) q[2];
ry(1.2438599288827135) q[3];
rz(-0.16823227051217582) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.4393868309949105) q[0];
rz(-1.0766692369393747) q[0];
ry(2.3629682294425254) q[1];
rz(0.1825521011516337) q[1];
ry(-2.758141448723783) q[2];
rz(1.2080945578817506) q[2];
ry(-0.6883986139408469) q[3];
rz(0.43872236090301286) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.0788068349730704) q[0];
rz(-2.730646547956349) q[0];
ry(1.296799543223793) q[1];
rz(-1.8402608713966409) q[1];
ry(1.0869612016661057) q[2];
rz(0.9497795044437011) q[2];
ry(1.8351995544807749) q[3];
rz(0.06269877331514366) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.6441741204588058) q[0];
rz(-1.6884599244432705) q[0];
ry(0.6568229652039178) q[1];
rz(3.1202251086897155) q[1];
ry(-0.25889866350062485) q[2];
rz(-2.339048210493949) q[2];
ry(0.008942236787976137) q[3];
rz(0.5440587236565247) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.07389631459687522) q[0];
rz(-0.042513063569018045) q[0];
ry(-1.6359623742152591) q[1];
rz(-1.047782713167448) q[1];
ry(2.1771300379938623) q[2];
rz(-1.4546319396826204) q[2];
ry(2.548129795720997) q[3];
rz(-1.8927507273873403) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.15920967198746547) q[0];
rz(0.21055397320119992) q[0];
ry(2.834726850227275) q[1];
rz(-3.110241450642852) q[1];
ry(-2.439723186758398) q[2];
rz(-1.0122842779360512) q[2];
ry(-2.1189242266107526) q[3];
rz(-0.8815145372440708) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.7406294249180732) q[0];
rz(-2.2295866130711044) q[0];
ry(0.6370635975468666) q[1];
rz(-1.9124054300704905) q[1];
ry(-0.09817155427367386) q[2];
rz(1.9444271903450119) q[2];
ry(1.480299421977504) q[3];
rz(-2.4110122076214804) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.33857848822947473) q[0];
rz(-2.0375958766797675) q[0];
ry(-0.07759068065311947) q[1];
rz(-2.268210268653409) q[1];
ry(-1.705897580219383) q[2];
rz(-0.51114411740301) q[2];
ry(-0.36162783454938735) q[3];
rz(2.4035687441721993) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.266864026545691) q[0];
rz(1.2921840099154929) q[0];
ry(-2.364388345477616) q[1];
rz(-0.3023425688901032) q[1];
ry(1.943865206115338) q[2];
rz(1.6083972371880455) q[2];
ry(-0.9747143840645709) q[3];
rz(-0.12749062826089474) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.6657368858738737) q[0];
rz(-0.20646388650246492) q[0];
ry(2.3347243968507003) q[1];
rz(-0.04220061929107021) q[1];
ry(0.36896552502934915) q[2];
rz(1.7498408019197322) q[2];
ry(2.9734846975517986) q[3];
rz(2.346979580289383) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.267830122315594) q[0];
rz(-1.2687358214784665) q[0];
ry(-1.84792589379059) q[1];
rz(-3.0825201850335415) q[1];
ry(-0.02304808526894539) q[2];
rz(-2.203333881780188) q[2];
ry(-0.732723800691849) q[3];
rz(-2.2746557404716388) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.1786172685026153) q[0];
rz(-0.4683057253119987) q[0];
ry(1.586094543479815) q[1];
rz(-0.014731201276712047) q[1];
ry(1.8382902217533585) q[2];
rz(0.6752068163632383) q[2];
ry(-0.33184000468933395) q[3];
rz(2.722650284851435) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.9557990770867334) q[0];
rz(0.7061634522754608) q[0];
ry(-0.8326322769451303) q[1];
rz(-0.44379200624211096) q[1];
ry(-0.4835332614734492) q[2];
rz(0.22629229993095024) q[2];
ry(0.7077916119039336) q[3];
rz(2.6252484436189145) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.373491588253881) q[0];
rz(3.140929160472432) q[0];
ry(-2.962211716796485) q[1];
rz(1.7325305981703258) q[1];
ry(-0.8351085520806629) q[2];
rz(-0.8539191388823808) q[2];
ry(2.5609164814086522) q[3];
rz(2.268595079286226) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.8679892839910672) q[0];
rz(-1.7452376877813691) q[0];
ry(-0.7392301000550701) q[1];
rz(2.4553416589144073) q[1];
ry(1.3438871275269635) q[2];
rz(1.526135897306001) q[2];
ry(-2.0663922300111546) q[3];
rz(-1.98105042921407) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.0916511678609524) q[0];
rz(-1.9697620872481105) q[0];
ry(0.8049345552394183) q[1];
rz(2.322430790622821) q[1];
ry(-0.879313115249638) q[2];
rz(0.15880222894547205) q[2];
ry(-1.9927730917048427) q[3];
rz(2.6098627792141893) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.30493842157033657) q[0];
rz(1.4119050847277888) q[0];
ry(0.020581419611693796) q[1];
rz(1.5814473971410337) q[1];
ry(-2.639753611951825) q[2];
rz(-2.3476750174193137) q[2];
ry(-0.4954641744350053) q[3];
rz(-2.3398596638086393) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.9563052541533157) q[0];
rz(-0.14151205352445692) q[0];
ry(-1.0370641651885215) q[1];
rz(-0.8482837421451256) q[1];
ry(0.23149415763933145) q[2];
rz(0.927408408059053) q[2];
ry(2.896490401194374) q[3];
rz(-1.4273924282417632) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.3642448149991098) q[0];
rz(-2.0103036096329765) q[0];
ry(-2.695910639460985) q[1];
rz(-1.8358314178531767) q[1];
ry(1.697065427335311) q[2];
rz(2.91877144494171) q[2];
ry(-0.17873954512285975) q[3];
rz(-1.793757674404374) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.4182231221253736) q[0];
rz(-1.5935712417262415) q[0];
ry(0.4791316396196642) q[1];
rz(-1.4934994769773913) q[1];
ry(-2.7771143119968897) q[2];
rz(-0.43992099733100787) q[2];
ry(-0.8500128171091905) q[3];
rz(-2.0865706036853364) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.3103213019361544) q[0];
rz(0.7917746637962031) q[0];
ry(-2.5951231501842456) q[1];
rz(2.807788422491598) q[1];
ry(-0.6010044884101132) q[2];
rz(2.755952939001208) q[2];
ry(0.7795795858327902) q[3];
rz(1.5341263559094616) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.9304092976279217) q[0];
rz(0.27121299672159616) q[0];
ry(-1.1795479614007656) q[1];
rz(0.9487450495108334) q[1];
ry(-0.9177415807773289) q[2];
rz(-1.3510381500239461) q[2];
ry(2.5677201552021742) q[3];
rz(2.800341607931096) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.1804734003758783) q[0];
rz(-2.1913107961282074) q[0];
ry(-1.820775850149547) q[1];
rz(-0.1288409111093345) q[1];
ry(1.9901912289980856) q[2];
rz(1.4396760501650676) q[2];
ry(-2.9317225039069243) q[3];
rz(2.4086941815605813) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.5113270155424603) q[0];
rz(0.8702627220446573) q[0];
ry(0.26086921019835874) q[1];
rz(2.7295567188393672) q[1];
ry(1.5866078775559274) q[2];
rz(0.1380392489492932) q[2];
ry(-0.9275639647941433) q[3];
rz(-0.8647521740422812) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.059777139140257056) q[0];
rz(-2.191376683219908) q[0];
ry(1.6116410357815227) q[1];
rz(-0.8334222086518431) q[1];
ry(0.8876816453740757) q[2];
rz(-2.0547618874146156) q[2];
ry(-2.7764809747450716) q[3];
rz(0.05007935573425338) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.6341115438654583) q[0];
rz(1.4255287353208885) q[0];
ry(1.2017563855882039) q[1];
rz(-2.5140021061197273) q[1];
ry(-0.5930916260436847) q[2];
rz(0.3987955916823341) q[2];
ry(-1.678254057277515) q[3];
rz(-0.9926149682979517) q[3];