# ==============================================================================
# app.R
# Interactive WDEI interface for the final Section 5 3D-printing case
# ==============================================================================
# ==============================================================================

options(stringsAsFactors = FALSE, warn = 1, scipen = 6, digits = 6)

interface_assert <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
}

INTERFACE_CFG <- list(
  result_path = Sys.getenv(
    "INTERFACE_RESULT_PATH",
    unset = "3D case results"
  ),
  interval_level = 0.95,
  show_reference_best = TRUE
)

interface_is_result <- function(x) {
  is.list(x) && !is.null(x$case) && !is.null(x$runs) &&
    !is.null(x$runs$WDEI) && !is.null(x$runs$WDEI$pred)
}

interface_validate_result <- function(result) {
  interface_assert(is.list(result) && !is.null(result$case),
             "The saved file is not a valid 3D-case result object.")
  interface_assert(!is.null(result$runs$WDEI),
             "The saved result does not contain the WDEI run.")
  case <- result$case
  run <- result$runs$WDEI
  required_case <- c("U", "U_unit", "t", "basis", "reference")
  required_pred <- c("muJ", "varJ", "integrated", "mu_curve", "var_curve")
  interface_assert(all(required_case %in% names(case)),
             "The saved result is missing required case objects.")
  interface_assert(all(required_pred %in% names(run$pred)),
             "The saved result is missing required WDEI predictions.")
  n <- nrow(case$U)
  interface_assert(n >= 1L && nrow(case$U_unit) == n,
             "The candidate input matrices are inconsistent.")
  interface_assert(nrow(case$reference$curves) == n &&
               length(case$reference$J) == n,
             "The reference environment does not match the candidate library.")
  interface_assert(nrow(run$pred$mu_curve) == n &&
               nrow(run$pred$var_curve) == n,
             "The WDEI predictions do not match the candidate library.")
  interface_assert(length(case$basis$q) == ncol(run$pred$mu_curve),
             "The functional integration weights do not match the curves.")
  invisible(TRUE)
}

interface_result_candidates <- function(cfg = INTERFACE_CFG) {
  path <- path.expand(cfg$result_path)
  direct <- unique(c(
    path,
    paste0(path, ".rds"),
    paste0(path, ".RDS"),
    paste0(path, ".RData"),
    paste0(path, ".rda")
  ))
  files <- direct[file.exists(direct) & !dir.exists(direct)]

  directories <- direct[dir.exists(direct)]
  if (length(directories)) {
    nested <- unlist(lapply(directories, function(directory) {
      list.files(directory, full.names = TRUE, recursive = TRUE,
                 all.files = FALSE, no.. = TRUE)
    }), use.names = FALSE)
    if (length(nested)) {
      nested <- nested[file.exists(nested) & !dir.exists(nested)]
      extension <- tolower(tools::file_ext(nested))
      likely <- extension %in% c("rds", "rdata", "rda", "")
      nested <- c(nested[likely], nested[!likely])
      files <- c(files, nested)
    }
  }
  unique(normalizePath(files, winslash = "/", mustWork = FALSE))
}

interface_try_read_file <- function(path) {
  object <- suppressWarnings(tryCatch(readRDS(path), error = function(e) NULL))
  if (interface_is_result(object)) return(object)

  workspace <- new.env(parent = baseenv())
  loaded <- suppressWarnings(tryCatch(
    load(path, envir = workspace), error = function(e) character()
  ))
  if (length(loaded)) {
    objects <- lapply(loaded, function(name) get(name, envir = workspace,
                                                 inherits = FALSE))
    valid <- which(vapply(objects, interface_is_result, logical(1)))
    if (length(valid)) return(objects[[valid[1L]]])
  }
  NULL
}

interface_result_in_session <- function() {
  names <- ls(envir = .GlobalEnv, all.names = TRUE)
  if (!length(names)) return(NULL)
  valid <- vapply(names, function(name) {
    interface_is_result(get(name, envir = .GlobalEnv, inherits = FALSE))
  }, logical(1))
  if (!any(valid)) return(NULL)
  name <- names[which(valid)[1L]]
  result <- get(name, envir = .GlobalEnv, inherits = FALSE)
  attr(result, "interface_source") <- paste("R session object", name)
  result
}

interface_load_result <- function(cfg = INTERFACE_CFG) {
  session_result <- interface_result_in_session()
  if (!is.null(session_result)) {
    interface_validate_result(session_result)
    return(session_result)
  }

  candidates <- interface_result_candidates(cfg)
  for (path in candidates) {
    result <- interface_try_read_file(path)
    if (!is.null(result)) {
      interface_validate_result(result)
      attr(result, "interface_source") <- path
      return(result)
    }
  }

  stop(paste0(
    "No complete WDEI result could be read from ", cfg$result_path, ". ",
    "The saved file must contain the complete result produced by 3D case.R, ",
    "including the case object and runs$WDEI."
  ), call. = FALSE)
}

interface_prepare_state <- function(result) {
  interface_validate_result(result)
  case <- result$case
  run <- result$runs$WDEI
  evaluated <- unique(run$idx_history)
  recommendation <- evaluated[which.min(run$pred$muJ[evaluated])]

  error <- run$pred$mu_curve - case$reference$curves
  q_matrix <- matrix(case$basis$q, nrow(error), length(case$basis$q), TRUE)
  candidate_rmse <- sqrt(rowSums(error^2 * q_matrix))
  local_error <- run$pred$mu_curve[recommendation, ] -
    case$reference$curves[recommendation, ]

  metrics <- c(
    Global_RMSE = mean(candidate_rmse),
    Integrated_Latent_Variance = mean(run$pred$integrated),
    POG = max(case$reference$J[recommendation] -
                min(case$reference$J), 0),
    Optimal_RMSE = sqrt(sum(case$basis$q * local_error^2))
  )

  if (!is.null(result$results)) {
    row <- result$results[result$results$Method == "WDEI", , drop = FALSE]
    if (nrow(row) == 1L) {
      comparison <- c(
        Global_RMSE = row$Global_Reference_RMSE,
        Integrated_Latent_Variance = row$Integrated_Latent_Variance,
        POG = row$Reference_POG,
        Optimal_RMSE = row$Optimal_RMSE
      )
      interface_assert(max(abs(metrics - comparison), na.rm = TRUE) < 1e-8,
                 "Recomputed WDEI indicators do not match the saved result table.")
    }
  }

  input_lower <- if (!is.null(case$cfg$input_lower)) {
    case$cfg$input_lower
  } else apply(case$U, 2, min)
  input_upper <- if (!is.null(case$cfg$input_upper)) {
    case$cfg$input_upper
  } else apply(case$U, 2, max)

  list(
    source = attr(result, "interface_source"),
    U = case$U,
    U_unit = case$U_unit,
    t = case$t,
    q = case$basis$q,
    input_lower = input_lower,
    input_upper = input_upper,
    reference_curves = case$reference$curves,
    reference_J = case$reference$J,
    reference_best_idx = case$reference$best_idx,
    idx_history = run$idx_history,
    recommendation = recommendation,
    metrics = metrics,
    pred = list(
      muJ = run$pred$muJ,
      varJ = run$pred$varJ,
      integrated = run$pred$integrated,
      mu_curve = run$pred$mu_curve,
      var_curve = run$pred$var_curve
    )
  )
}

interface_nearest_candidate <- function(values, state) {
  values <- as.numeric(values)
  interface_assert(length(values) == 3L && all(is.finite(values)),
             "All three printing parameters must be finite numbers.")
  unit <- (values - state$input_lower) /
    pmax(state$input_upper - state$input_lower, 1e-12)
  distance2 <- rowSums((state$U_unit -
    matrix(unit, nrow(state$U_unit), 3L, byrow = TRUE))^2)
  which.min(distance2)
}

interface_metrics_at <- function(index, state) {
  local_error <- state$pred$mu_curve[index, ] -
    state$reference_curves[index, ]
  c(
    Global_RMSE = unname(state$metrics["Global_RMSE"]),
    Integrated_Latent_Variance = unname(
      state$metrics["Integrated_Latent_Variance"]
    ),
    POG = max(state$reference_J[index] - min(state$reference_J), 0),
    Optimal_RMSE = sqrt(sum(state$q * local_error^2))
  )
}

interface_selection <- function(requested, state) {
  index <- interface_nearest_candidate(requested, state)
  list(
    index = index,
    requested = as.numeric(requested),
    candidate = state$U[index, ],
    evaluated = index %in% state$idx_history,
    metrics = interface_metrics_at(index, state)
  )
}

interface_metric_card <- function(label, value) {
  shiny::tags$div(
    class = "metric-card",
    shiny::tags$div(class = "metric-label", label),
    shiny::tags$div(
      class = "metric-value",
      formatC(as.numeric(value), format = "f", digits = 6)
    )
  )
}

interface_recommended_parameter <- function(label, value, unit, digits) {
  shiny::tags$div(
    class = "recommended-item",
    shiny::tags$div(class = "recommended-label", label),
    shiny::tags$div(
      class = "recommended-value",
      paste0(formatC(as.numeric(value), format = "f", digits = digits), " ", unit)
    )
  )
}

interface_parameter_control <- function(input_id, label, unit, minimum,
                                        maximum, value, step, digits) {
  shiny::tags$div(
    class = "parameter-control",
    shiny::tags$div(
      class = "parameter-heading",
      shiny::tags$span(label),
      shiny::tags$span(class = "parameter-unit", unit)
    ),
    shiny::sliderInput(
      input_id, label = NULL, min = minimum, max = maximum,
      value = value, step = step, ticks = FALSE, sep = ""
    ),
    shiny::tags$div(
      class = "parameter-range",
      paste0("Range  ", formatC(minimum, format = "f", digits = digits),
             "  to  ", formatC(maximum, format = "f", digits = digits))
    )
  )
}

interface_build_app <- function(state, cfg = INTERFACE_CFG) {
  interface_assert(requireNamespace("shiny", quietly = TRUE),
             "Package 'shiny' is required. Install it with install.packages('shiny').")

  rec <- state$recommendation
  default <- state$U[rec, ]
  step <- c(0.001, 0.001, 0.1)

  ui <- shiny::fluidPage(
    shiny::tags$head(
      shiny::tags$style(shiny::HTML("\
        body { margin: 0; background: #f3f6f9; color: #202832;\
          font-family: Arial, sans-serif; }\
        .container-fluid { width: 100%; max-width: 1580px;\
          margin: 0 auto; padding: 22px 30px 30px; }\
        .app-title { margin: 0 0 20px; color: #173b5e;\
          font-family: 'Times New Roman', serif; font-size: 30px;\
          font-weight: 700; line-height: 1.2; }\
        .dashboard-grid { display: grid;\
          grid-template-columns: minmax(310px, 340px) minmax(0, 1fr);\
          gap: 22px; align-items: start; }\
        .control-panel, .plot-panel, .metric-card { background: #ffffff;\
          border: 1px solid #dbe3ea; box-shadow: 0 3px 12px rgba(28,51,73,.06); }\
        .control-panel { border-radius: 10px; padding: 24px;\
          position: sticky; top: 18px; }\
        .control-title { margin: 0 0 5px; color: #173b5e;\
          font-size: 20px; font-weight: 700; }\
        .control-rule { height: 2px; width: 44px; margin: 0 0 24px;\
          background: #2b83b8; border-radius: 2px; }\
        .parameter-control { margin-bottom: 25px; }\
        .parameter-heading { display: flex; justify-content: space-between;\
          align-items: baseline; gap: 10px; margin-bottom: 4px;\
          color: #273746; font-size: 15px; font-weight: 700; }\
        .parameter-unit { color: #71808e; font-size: 12px; font-weight: 500; }\
        .parameter-control .form-group { margin-bottom: 0; }\
        .parameter-range { margin-top: 3px; color: #7b8793;\
          font-size: 11px; text-align: right; }\
        .irs--shiny .irs-bar { background: #2b83b8; border-color: #2b83b8; }\
        .irs--shiny .irs-single { background: #2b83b8; }\
        .irs--shiny .irs-single:before { border-top-color: #2b83b8; }\
        .update-wrap { margin-top: 8px; }\
        .update-wrap .btn { width: 100%; min-height: 52px;\
          border-radius: 7px; font-size: 17px; font-weight: 700;\
          letter-spacing: .2px; }\
        .update-wrap .btn-primary { background: #176fa6; border-color: #176fa6;\
          box-shadow: 0 3px 8px rgba(23,111,166,.20); }\
        .update-wrap .btn-primary:hover,\
        .update-wrap .btn-primary:focus { background: #125d8d;\
          border-color: #125d8d; }\
        .main-content { min-width: 0; }\
        .metric-grid { display: grid;\
          grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }\
        .metric-card { border-left: 5px solid #2784b8; border-radius: 8px;\
          padding: 16px 17px; min-height: 112px; margin: 0;\
          display: flex; flex-direction: column; justify-content: space-between; }\
        .metric-label { color: #596775; font-size: 13px;\
          line-height: 1.3; min-height: 34px; }\
        .metric-value { color: #173b5e; font-family: 'Times New Roman', serif;\
          font-size: 27px; font-weight: 700; line-height: 1.1; }\
        .recommended-panel { margin-top: 24px; padding-top: 20px;\
          border-top: 1px solid #dbe3ea; }\
        .recommended-heading { margin: 0 0 12px; color: #173b5e;\
          font-size: 16px; font-weight: 700; line-height: 1.3; }\
        .recommended-row { display: grid; grid-template-columns: 1fr; gap: 9px; }\
        .recommended-item { display: flex; align-items: baseline;\
          justify-content: space-between; gap: 12px; padding: 10px 13px;\
          background: #f4f8fb; border: 1px solid #d9e6ef; border-radius: 6px; }\
        .recommended-label { color: #5a6875; font-size: 13px; }\
        .recommended-value { color: #173b5e; font-family: 'Times New Roman', serif;\
          font-size: 19px; font-weight: 700; white-space: nowrap; }\
        .plot-panel { margin-top: 18px; border-radius: 10px; padding: 18px 20px 8px;\
          overflow: hidden; }\
        @media (max-width: 1180px) {\
          .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }\
        }\
        @media (max-width: 860px) {\
          .container-fluid { padding: 18px; }\
          .dashboard-grid { grid-template-columns: 1fr; }\
          .control-panel { position: static; }\
          .metric-grid { margin-top: 2px; }\
        }\
        @media (max-width: 560px) {\
          .metric-grid { grid-template-columns: 1fr; }\
        }\
      "))
    ),
    shiny::tags$h1(
      class = "app-title", "WDEI Interactive Prediction for 3D Printing"
    ),
    shiny::div(
      class = "dashboard-grid",
      shiny::div(
        class = "control-panel",
        shiny::tags$h2(class = "control-title", "Printing parameters"),
        shiny::div(class = "control-rule"),
        interface_parameter_control(
          "layer", "Layer thickness", "mm",
          state$input_lower[1], state$input_upper[1], default[1], step[1], 3
        ),
        interface_parameter_control(
          "infill", "Infilling rate", "proportion",
          state$input_lower[2], state$input_upper[2], default[2], step[2], 3
        ),
        interface_parameter_control(
          "speed", "Printing speed", "mm/s",
          state$input_lower[3], state$input_upper[3], default[3], step[3], 1
        ),
        shiny::div(
          class = "update-wrap",
          shiny::actionButton(
            "update", "Update WDEI results", class = "btn-primary"
          )
        ),
        shiny::div(
          class = "recommended-panel",
          shiny::tags$h2(
            class = "recommended-heading", "Recommended WDEI parameters"
          ),
          shiny::div(
            class = "recommended-row",
            interface_recommended_parameter(
              "Layer thickness", default[1], "mm", 3
            ),
            interface_recommended_parameter(
              "Infilling rate", default[2], "", 3
            ),
            interface_recommended_parameter(
              "Printing speed", default[3], "mm/s", 2
            )
          )
        )
      ),
      shiny::div(
        class = "main-content",
        shiny::div(
          class = "metric-grid",
          shiny::uiOutput("metric_global_rmse"),
          shiny::uiOutput("metric_iv"),
          shiny::uiOutput("metric_pog"),
          shiny::uiOutput("metric_optimal_rmse")
        ),
        shiny::div(
          class = "plot-panel",
          shiny::plotOutput("wdei_plot", width = "100%", height = "610px")
        )
      )
    )
  )

  server <- function(input, output, session) {
    current <- shiny::reactiveVal(interface_selection(default, state))

    shiny::observeEvent(input$update, {
      current(interface_selection(c(input$layer, input$infill, input$speed), state))
    })

    output$metric_global_rmse <- shiny::renderUI({
      interface_metric_card("Global RMSE", current()$metrics["Global_RMSE"])
    })
    output$metric_iv <- shiny::renderUI({
      interface_metric_card(
        "Integrated latent variance",
        current()$metrics["Integrated_Latent_Variance"]
      )
    })
    output$metric_pog <- shiny::renderUI({
      interface_metric_card("POG at selected input", current()$metrics["POG"])
    })
    output$metric_optimal_rmse <- shiny::renderUI({
      interface_metric_card(
        "Optimal RMSE at selected input",
        current()$metrics["Optimal_RMSE"]
      )
    })

    output$wdei_plot <- shiny::renderPlot({
      item <- current()
      index <- item$index
      time <- state$t
      posterior_mean <- state$pred$mu_curve[index, ]
      posterior_sd <- sqrt(pmax(state$pred$var_curve[index, ], 0))
      z <- stats::qnorm(0.5 + cfg$interval_level / 2)
      lower_curve <- posterior_mean - z * posterior_sd
      upper_curve <- posterior_mean + z * posterior_sd
      reference_curve <- state$reference_curves[index, ]
      best_curve <- state$reference_curves[state$reference_best_idx, ]
      limits <- range(lower_curve, upper_curve, reference_curve, best_curve,
                      finite = TRUE)

      old <- graphics::par(no.readonly = TRUE)
      on.exit(graphics::par(old), add = TRUE)
      graphics::par(
        family = "serif", mar = c(4.5, 4.8, 3.4, 1.0),
        mgp = c(2.8, 0.8, 0), tcl = -0.25, las = 1
      )
      graphics::plot(
        time, posterior_mean, type = "n", ylim = limits,
        xlab = "Time (s)", ylab = "Functional response",
        main = sprintf(
          "WDEI functional response prediction\nu = (%.3f, %.3f, %.2f)",
          item$candidate[1], item$candidate[2], item$candidate[3]
        ),
        cex.lab = 1.15, cex.main = 1.05
      )
      graphics::grid(col = "#D9D9D9", lty = 3)
      graphics::polygon(
        c(time, rev(time)), c(lower_curve, rev(upper_curve)),
        col = "#E3E3E3", border = NA
      )
      if (isTRUE(cfg$show_reference_best)) {
        graphics::lines(time, best_curve, col = "black", lty = 3, lwd = 2.2)
      }
      graphics::lines(time, reference_curve, col = "#0072B2",
                      lty = 1, lwd = 2.2)
      graphics::lines(time, posterior_mean, col = "#D55E00",
                      lty = 2, lwd = 2.2)
      graphics::box()
      graphics::legend(
        "bottomright",
        legend = c(
          "Reference best feasible curve",
          "Reference curve at selected input",
          "WDEI GPFR posterior mean",
          paste0(round(100 * cfg$interval_level),
                 "% latent mean interval")
        ),
        col = c("black", "#0072B2", "#D55E00", "#E3E3E3"),
        lty = c(3, 1, 2, NA), lwd = c(2.2, 2.2, 2.2, NA),
        pch = c(NA, NA, NA, 15), pt.cex = 1.7,
        bty = "n", cex = 0.85
      )
    }, res = 110)
  }

  shiny::shinyApp(ui = ui, server = server)
}

if (identical(Sys.getenv("INTERFACE_SKIP_APP", unset = "0"), "1")) {
  NULL
} else {
  Interactive_Result <- interface_load_result(INTERFACE_CFG)
  Interactive_State <- interface_prepare_state(Interactive_Result)
  Interactive_App <- interface_build_app(Interactive_State, INTERFACE_CFG)
  Interactive_App
}
