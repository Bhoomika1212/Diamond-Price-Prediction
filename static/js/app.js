const form =
    document.getElementById(
        "predictionForm"
    );


const button =
    document.getElementById(
        "predictBtn"
    );


const price =
    document.getElementById(
        "price"
    );


const resultText =
    document.getElementById(
        "resultText"
    );


const errorBox =
    document.getElementById(
        "errorBox"
    );



form.addEventListener(
    "submit",
    async function(event) {

        event.preventDefault();


        errorBox.textContent = "";


        button.disabled = true;


        button.style.opacity = "0.65";


        button
            .querySelector(
                "span:first-child"
            )
            .textContent =
            "Calculating...";


        try {


            const formData =
                new FormData(form);


            const data =
                Object.fromEntries(
                    formData.entries()
                );


            const response =
                await fetch(
                    "/predict",
                    {

                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify(data)

                    }
                );


            const result =
                await response.json();


            if (
                !response.ok ||
                !result.success
            ) {

                throw new Error(
                    result.error ||
                    "Prediction failed."
                );

            }


            price.textContent =
                new Intl.NumberFormat(
                    "en-US",
                    {

                        style: "currency",

                        currency: "USD",

                        maximumFractionDigits: 0

                    }
                ).format(
                    result.price
                );


            resultText.textContent =
                "Estimated from the supplied diamond characteristics using the trained Random Forest model.";


        }


        catch(error) {


            errorBox.textContent =
                error.message;


            price.textContent =
                "$—";


            resultText.textContent =
                "Please correct the input values and try again.";

        }


        finally {


            button.disabled = false;


            button.style.opacity = "1";


            button
                .querySelector(
                    "span:first-child"
                )
                .textContent =
                "Estimate Diamond Price";

        }

    }
);