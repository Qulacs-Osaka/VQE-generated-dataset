OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(1.5994049698926558) q[0];
rz(1.1585463439088421) q[0];
ry(-0.05735805478988975) q[1];
rz(-0.636896139481319) q[1];
ry(1.547560709857203) q[2];
rz(1.2729883997169402) q[2];
ry(1.5707917789908565) q[3];
rz(-1.5755301035133797) q[3];
ry(1.5707838618654661) q[4];
rz(0.3311212919647639) q[4];
ry(-1.507543630775407) q[5];
rz(0.0003084918667031218) q[5];
ry(3.141583328184279) q[6];
rz(3.0409095389087017) q[6];
ry(-1.5035770923784215) q[7];
rz(1.4104424953972279) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(3.1411739015482683) q[0];
rz(2.9849516725157637) q[0];
ry(-0.002308905947265849) q[1];
rz(0.6372682625665788) q[1];
ry(-0.6019365194458607) q[2];
rz(-2.1239974414728064) q[2];
ry(-0.8699480631268424) q[3];
rz(2.8818262485730934) q[3];
ry(3.0815292456159935) q[4];
rz(-1.7096929990831855) q[4];
ry(-1.5708178328346352) q[5];
rz(0.24991507642407035) q[5];
ry(1.7151338642797982) q[6];
rz(0.6331107191233887) q[6];
ry(1.4122804416024737) q[7];
rz(-1.392540917210705) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(0.11147131768260349) q[0];
rz(-2.1367756841654986) q[0];
ry(-0.08347770152818454) q[1];
rz(1.923041466061493) q[1];
ry(1.4153135702521524e-05) q[2];
rz(0.800974554061765) q[2];
ry(3.1415855430388593) q[3];
rz(2.8164585517718157) q[3];
ry(0.06269786738488084) q[4];
rz(2.653400206549217) q[4];
ry(0.05661324448986818) q[5];
rz(1.2147710398268394) q[5];
ry(0.38323508622256913) q[6];
rz(-0.6638919416216001) q[6];
ry(-0.06254336021929774) q[7];
rz(-0.0326824812164217) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.6829159137546197) q[0];
rz(1.600849782125823) q[0];
ry(0.6439452592236403) q[1];
rz(-3.1391850253985365) q[1];
ry(1.4044500637688833) q[2];
rz(-2.2620910264649594) q[2];
ry(0.8710257623195146) q[3];
rz(-0.5885111744518863) q[3];
ry(-0.01822376048844768) q[4];
rz(-2.3673443582360787) q[4];
ry(0.0003341029185767477) q[5];
rz(-0.14783073405289507) q[5];
ry(-1.8395708951963794) q[6];
rz(0.006525777717373598) q[6];
ry(3.0231322396702467) q[7];
rz(-0.06201647674778476) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(0.0003504427893351405) q[0];
rz(-2.8350122661309136) q[0];
ry(-1.5752593003049595) q[1];
rz(2.889726860275162) q[1];
ry(3.048406285700926) q[2];
rz(0.26252610340741017) q[2];
ry(-0.5470140283015245) q[3];
rz(0.03328004788323912) q[3];
ry(0.9779298777542429) q[4];
rz(0.4291168869243344) q[4];
ry(-2.9301296234608922) q[5];
rz(3.091112324440181) q[5];
ry(1.8756258469124556) q[6];
rz(1.1086142227745341) q[6];
ry(3.087702823012499) q[7];
rz(-0.11485254471784313) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.5576817266936975) q[0];
rz(-3.0039120591911086) q[0];
ry(0.6572385877058011) q[1];
rz(-1.1752073970637107) q[1];
ry(-0.009145707111063777) q[2];
rz(3.1372228586592605) q[2];
ry(0.005364290378154557) q[3];
rz(-1.1939969015281298) q[3];
ry(-3.1294424050819543) q[4];
rz(0.3787539179738735) q[4];
ry(-0.010682177451829311) q[5];
rz(3.0966224622207954) q[5];
ry(-2.8719752835374734) q[6];
rz(2.7586924830556705) q[6];
ry(-1.6064467548948074) q[7];
rz(-1.3770375614715311) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.8670193635616632) q[0];
rz(-2.1364012028223742) q[0];
ry(-0.1351230033006381) q[1];
rz(2.3425586859463228) q[1];
ry(1.437312294432099) q[2];
rz(2.537230146113954) q[2];
ry(-1.9112210314523908) q[3];
rz(2.0660915095087855) q[3];
ry(2.0508079739970433) q[4];
rz(-2.432244972977811) q[4];
ry(3.0264295290257417) q[5];
rz(0.6994985809273179) q[5];
ry(-0.24359156104481006) q[6];
rz(-2.4839859389004855) q[6];
ry(2.899239498832087) q[7];
rz(-2.148421511053043) q[7];