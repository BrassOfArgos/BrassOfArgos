class FederalGovernment:
    def __init__(self, name):
        self.name = name
        self.branches = []

    def add_branch(self, branch):
        self.branches.append(branch)


class Branch:
    def __init__(self, name, budget):
        self.name = name
        self.budget = budget
        self.departments = []

    def add_department(self, department):
        self.departments.append(department)


class Department:
    def __init__(self, name, budget, purpose):
        self.name = name
        self.budget = budget
        self.purpose = purpose

    def display_info(self):
        print(f"{self.name} Department:")
        print(f"Budget: ${self.budget}")
        print(f"Purpose: {self.purpose}")
        print("\n")


# Creating instances of the classes
government = FederalGovernment("United States Federal Government")

executive_branch = Branch("Executive Branch", 50000000000)  # Example budget
legislative_branch = Branch("Legislative Branch", 30000000000)  # Example budget
judicial_branch = Branch("Judicial Branch", 10000000000)  # Example budget

# Adding departments to the branches
executive_branch.add_department(Department("Department of Defense", 7000000000, "National Defense"))
executive_branch.add_department(Department("Department of Health and Human Services", 8000000000, "Public Health"))
legislative_branch.add_department(Department("Congressional Budget Office", 2000000000, "Budget Analysis"))
judicial_branch.add_department(Department("Supreme Court", 500000000, "Interpreting the Constitution"))

# Adding branches to the government
government.add_branch(executive_branch)
government.add_branch(legislative_branch)
government.add_branch(judicial_branch)

# Displaying information
print(f"{government.name} Budget Overview:\n")
for branch in government.branches:
    print(f"{branch.name} Budget: ${branch.budget}")
    for department in branch.departments:
        department.display_info()
