module solve (
    input  logic clk,
    input  logic start,
    output logic done,
    output logic [31:0] A_addr,
    input  logic [31:0] A_data,
    output logic [31:0] B_addr,
    input  logic [31:0] B_data,
    output logic [31:0] C_addr,
    output logic [31:0] C_data,
    output logic C_write,
    input  logic signed [31:0] N
);

    // A_data and B_data hold the elements at the addresses of the previous cycle, as raw binary32 bits.
    // Write C[C_addr] = C_data on every cycle C_write is high. Registers are zero before start.
    assign A_addr = 0;
    assign B_addr = 0;
    assign C_addr = 0;
    assign C_data = 0;
    assign C_write = 1'b0;
    assign done = 1'b0;

endmodule
