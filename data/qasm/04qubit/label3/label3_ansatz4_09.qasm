OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.0210044460890657) q[0];
rz(-2.921778945669764) q[0];
ry(2.52767947111435) q[1];
rz(2.8120525292594367) q[1];
ry(0.3657741057629165) q[2];
rz(-2.2522005490106003) q[2];
ry(-0.5575244639380892) q[3];
rz(2.9480850946899384) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.5058063770858006) q[0];
rz(-0.38848619347080326) q[0];
ry(2.3261327209841003) q[1];
rz(1.9088916195813486) q[1];
ry(1.699753513934871) q[2];
rz(1.2815520724417215) q[2];
ry(0.8299687986608904) q[3];
rz(-0.8241249312721228) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.002018665103253) q[0];
rz(1.6901344103969635) q[0];
ry(-1.2818483048175366) q[1];
rz(-1.3342902582821055) q[1];
ry(1.2485099041170447) q[2];
rz(1.4030436218711388) q[2];
ry(2.1943667511307146) q[3];
rz(1.8340978161925996) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.7871265896078246) q[0];
rz(-0.2978827518794524) q[0];
ry(2.5347042307248953) q[1];
rz(-0.4332420133989879) q[1];
ry(-2.120612210870897) q[2];
rz(1.869954869693712) q[2];
ry(-1.3846329420103078) q[3];
rz(-0.40156808615568895) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.615994234226151) q[0];
rz(0.4806770411554817) q[0];
ry(-0.9823715276221451) q[1];
rz(-1.2069287821762744) q[1];
ry(-0.9166749610390577) q[2];
rz(-2.5800744262737547) q[2];
ry(-1.9174218816834143) q[3];
rz(2.6330822305213033) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.3793447449738833) q[0];
rz(0.2721755373011546) q[0];
ry(2.0432350720544608) q[1];
rz(-3.117974909589936) q[1];
ry(2.985664452838241) q[2];
rz(-1.9983842535552903) q[2];
ry(-1.4136505451509223) q[3];
rz(2.7043667897284394) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(3.117278092159482) q[0];
rz(-2.2071844225869652) q[0];
ry(2.335034402503741) q[1];
rz(0.6540706519070615) q[1];
ry(-2.5513098361395206) q[2];
rz(-0.42303382045512494) q[2];
ry(-1.2148461564548212) q[3];
rz(-2.5305912088220817) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.4534391244061937) q[0];
rz(1.4083557200682877) q[0];
ry(3.0475375485051255) q[1];
rz(2.0190286373963464) q[1];
ry(-2.7114658417496282) q[2];
rz(0.7415329797052346) q[2];
ry(0.14166868416366185) q[3];
rz(1.7922629888960495) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.1614822811417687) q[0];
rz(-0.06910363659799106) q[0];
ry(-0.23676617200601535) q[1];
rz(-2.9917225180009734) q[1];
ry(2.919937312243952) q[2];
rz(-0.5547544209370561) q[2];
ry(-1.951140508574221) q[3];
rz(1.2602181087732036) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.996059100959071) q[0];
rz(-2.7781541994884886) q[0];
ry(-2.412688403274543) q[1];
rz(-1.1630717755294808) q[1];
ry(3.131522607649936) q[2];
rz(-1.7585638637014729) q[2];
ry(-2.142726433826704) q[3];
rz(2.4297607401277106) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.05694816793173629) q[0];
rz(-1.499568609083025) q[0];
ry(-1.6445625828078922) q[1];
rz(0.33881001364341473) q[1];
ry(0.3629919296285998) q[2];
rz(0.35863845109479) q[2];
ry(-3.0301138668535055) q[3];
rz(-0.7534966881361519) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-3.0713710595314963) q[0];
rz(-2.9438078928262503) q[0];
ry(-3.017187844085774) q[1];
rz(1.3756633430851617) q[1];
ry(0.9309793529229858) q[2];
rz(-0.6184011554348191) q[2];
ry(3.0682274352476946) q[3];
rz(-2.9073860014450004) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.13211096698271607) q[0];
rz(-2.9030658754160603) q[0];
ry(-2.330469554206299) q[1];
rz(-1.9994767852678015) q[1];
ry(1.4667683994844287) q[2];
rz(-2.040576242302622) q[2];
ry(2.20037752150469) q[3];
rz(-1.13651803877291) q[3];