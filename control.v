module control1(clk, reset, addr_i, addr_w, wr_addr_f, rd_addr_f, we_f, wr_addr_out, we_f_out, control, init, finish, status, select);
	
    input             				clk, reset;
    input        [31:0]             control, init, finish;
    output wire	 [12:0] 			addr_i;
    output wire  [9:0] 				addr_w; 
    output reg   [11:0] 			wr_addr_f;
    output reg   [11:0]  			rd_addr_f;
    output reg   [11:0]             wr_addr_out;
    output reg   [3:0] 				we_f;
    output reg   [3:0]              we_f_out;
    output reg 	 [31:0]				status;
	output reg	    				select;
	
	// Wires and registers for control circuit
    reg 	[1:0]   				state;
	reg		[11:0]					addr_f_reg;
	wire	[3:0]					we_f_reg, we_f_out_reg;
    reg 	[3:0]   				we_f_reg_1, we_f_out_reg_1, we_f_out_reg_2;
	wire	[31:0]					status_reg; 
	wire                            select_reg;
	reg     [31:0]					status_reg_1;
	reg                             select_reg_1;
	reg 	[2:0]					tr, tc;
	reg		[3:0]					i, j;
	
	// State register
    always @(posedge clk) begin
        if (reset)
          state <= 0;
        else begin
            case(state)
            0: begin
                if(control == 1)
                    state <= 1;
                else
                    state <= 0;
				tr <= 0;
				tc <= 0;
				i <= 0;
				j <= 0;
				addr_f_reg <= 0;
            end
            1: begin
				if(tc != 7) begin
					tc <= tc + 1;
					addr_f_reg <= 4 * ((8 * tr) + (tc + 1));
				end
				else begin
					tc <= 0;
					if(tr != 7) begin
						tr <= tr + 1;
						addr_f_reg <= 4 * (8 * (tr + 1));
					end
					else begin
						tr <= 0;
						addr_f_reg <= 0;
						if(j != 2) begin
							j <= j + 1;
						end
						else begin
							j <= 0;
							if(i != 2) begin
								i <= i + 1;
							end
							else begin
								i <= 0;
								if(finish == 1)
								    state <= 2;
								else
								    state <= 3;
							end
						end
					end
				end
            end
            2: begin
                if(tc != 7) begin
                    tc <= tc + 1;
                    addr_f_reg <= 4 * ((8 * tr) + (tc + 1));
                 end
                 else begin
                    tc <= 0;
                    if(tr != 7) begin
                        tr <= tr + 1;
                        addr_f_reg <= 4 * 8 * (tr + 1);
                    end
                    else begin
                        tr <= 0;
                        state <= 3;
                        addr_f_reg <= 0;
                    end
                 end
            end
            3: begin
                if(control == 0)
                    state <= 0;
                else
                    state <= 3;
            end
            endcase
        end
        we_f_out <= we_f_out_reg_2;
        we_f_out_reg_2 <= we_f_out_reg_1;
        we_f_out_reg_1 <= we_f_out_reg;
		we_f <= we_f_reg_1;
		we_f_reg_1 <= we_f_reg;
		wr_addr_out <= wr_addr_f;
        wr_addr_f <= rd_addr_f;
        rd_addr_f <= addr_f_reg;
        status <= status_reg_1;
        status_reg_1 <= status_reg; 
		select <= select_reg_1;
		select_reg_1 <= select_reg; 
    end 


    // Assign output and address values
	assign addr_i = 4 * ((10 * (tr + i)) + (tc + j));
	assign addr_w = 4 * ((3 * i) + j);
    assign we_f_reg = (state == 1) ? 4'hf : 0;
    assign we_f_out_reg = (state == 2) ? 4'hf : 0;
    assign status_reg = (state == 3) ? 32'd1 : 32'd0;
	assign select_reg = ((state == 1) & (i == 0) & (j == 0) & (init == 1)) ? 1 : 0;
	
endmodule

module testControl();
	
	reg             		clk, reset;
	reg	   [31:0]			control, init, finish;
    wire   [12:0] 		    addr_i;
    wire   [9:0] 			addr_w; 
    wire   [11:0] 			wr_addr_f;
    wire   [11:0]  			rd_addr_f;
    wire   [11:0]           wr_addr_out;
    wire   [3:0] 			we_f;
    wire   [3:0]            we_f_out;
    wire   [31:0]			status;
	wire					select;
	
	control1 ctrl(clk, reset, addr_i, addr_w, wr_addr_f, rd_addr_f, we_f, wr_addr_out, we_f_out, control, init, finish, status, select);

	initial clk = 0;
	always #5 clk = ~clk;
	
	initial begin
		reset = 1;
		control = 0;
		init = 0;
		finish = 1;
		
		@(posedge clk);
		#1; reset = 0;

		@(posedge clk);
		#1; control = 1;
		
		wait(status);
		@(posedge clk);
		#1; control = 0;
		
		#10;
		$finish;
	end

endmodule
