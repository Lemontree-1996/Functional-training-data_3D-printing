


library(shiny)
library(DT)


# Load already saved results (e.g. selected_results.RData)
load("selected_results.RData")

# UI
ui <- fluidPage(
  titlePanel("3D Printing Parameter Optimization"),
  
  sidebarLayout(
    sidebarPanel(
      #
      selectInput("Layer_thickness", "Layer_thickness (mm):", choices = unique(scalar_test1[,1])),
      selectInput("Infilling_rate", "Infilling_rate:", choices = unique(scalar_test1[,2])),
      selectInput("Printing_speed", "Printing_speed (mm/s):", choices = unique(scalar_test1[,3])),
      
      # Calculate button
      actionButton("calc_btn", "Evaluation indicators"),
      # 
      verbatimTextOutput("rmse_value"),
      verbatimTextOutput("variance_value"),
      verbatimTextOutput("cp_value"),
      #
      verbatimTextOutput("selected_row")
    ),
    
    mainPanel(
      #
      div(style = "text-align: center;", 
          plotOutput("optimization_plot", width = "100%")),
      #
      DTOutput("parameters_table")
    )
  )
)


#
server <- function(input, output, session) {
  
  observeEvent(input$calc_btn, {
    #
    index <- which(
      round(scalar_test1[,1], 2) == round(as.numeric(input$Layer_thickness), 2) &
        round(scalar_test1[,2], 2) == round(as.numeric(input$Infilling_rate), 2) &
        round(scalar_test1[,3], 2) == round(as.numeric(input$Printing_speed), 2)
    )
    #
    if (length(index) == 0) {
      output$selected_row <- renderText({ "⚠️ The entered parameter is not in the grid data, please select it again!" })
      return()
    }
    
    #
    output$selected_row <- renderText({ paste("Parameter Index: ", index) })
    
    #
    real_curve <- F_star      
    predicted_curve <- predicted_means1[index, ]  
    predicted_sd <- predicted_sds1[index, ]      
    
    # RMSE
    rmse <- sqrt(mean((predicted_curve - real_curve)^2))
    
    # S^2
    variance <- mean(predicted_sd^2)
    
    # CP
    upper_bound <- predicted_curve + 1.96 * predicted_sd
    lower_bound <- predicted_curve - 1.96 * predicted_sd
    cp_value <- mean(real_curve >= lower_bound & real_curve <= upper_bound)
    
    # Indicators
    output$rmse_value <- renderText({ paste("RMSE: ", round(rmse, 3)) })
    output$variance_value <- renderText({ paste("S²: ", round(variance, 3)) })
    output$cp_value <- renderText({ paste("CP: ", round(cp_value, 3)) })
    
    #
    output$optimization_plot <- renderPlot({
      par(mfrow = c(1, 1), mar = c(5, 5, 1, 1))
      
      #
      plot.new()
      dev.size(units = "in")
      dev.control("in")
      par(mai = c(1, 1, 0.5, 0.5))
      
      plot(seq(1, 40), real_curve, type = "l", col = "red", lwd = 2, ylim = c(0,6),family = "serif",
           xlab = "Time (s)", ylab = "Force (kN)", main = "", axes = FALSE, cex.lab = 1.5, cex.axis = 1.3)
      
      #
      axis(1, at = seq(1, 40, length.out = 5), labels = c(0, 5, 10, 15, 20), cex.axis = 1.3)
      axis(2, cex.axis = 1.3)
      box()
      
      # 95% prediction interval
      polygon(c(seq(1, 40), rev(seq(1, 40))), 
              c(upper_bound, rev(lower_bound)), 
              col = rgb(0.6, 0.3, 0.8, alpha = 0.25), border = NA)
      #
      lines(seq(1, 40), predicted_curve, col = "blue", lwd = 2, lty = 2)
      #
      legend("bottomright", legend = c("Objective curve", "Predictive curve", "95% Prediction Interval"), 
             cex = 1, col = c("red", "blue", rgb(0.6, 0.3, 0.8, alpha = 0.25)), 
             lty = c(1, 2, NA), lwd = c(2, 2, NA), fill = c(NA, NA, rgb(0.6, 0.3, 0.8, alpha = 0.25)), border = NA)
      
      grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted", lwd = 0.8)
    }, width = 410, height = 410)
    
    #
    output$parameters_table <- renderDT({
      datatable(data.frame(scalar_test1[index, ]), options = list(pageLength = 5, autoWidth = TRUE))
    })
  })
}

#  Shiny
shinyApp(ui = ui, server = server)



