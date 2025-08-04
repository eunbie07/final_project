import React, { useState } from "react";

const mockHouses = [
  {
    id: 1,
    address: "123 Main St, Anytown USA",
    beds: 3,
    baths: 2,
    area: 1500, // sqft
    style: "Modern",
    image: "https://via.placeholder.com/300x200?text=Modern+House",
  },
  {
    id: 2,
    address: "456 Oak Ave, Anytown USA",
    beds: 4,
    baths: 3,
    area: 2200,
    style: "Vintage",
    image: "https://via.placeholder.com/300x200?text=Vintage+House",
  },
  {
    id: 3,
    address: "789 Pine Ln, Anytown USA",
    beds: 2,
    baths: 1,
    area: 900,
    style: "Minimalist",
    image: "https://via.placeholder.com/300x200?text=Minimalist+House",
  },
  {
    id: 4,
    address: "101 Maple Dr, Anytown USA",
    beds: 3,
    baths: 2,
    area: 1800,
    style: "Industrial",
    image: "https://via.placeholder.com/300x200?text=Industrial+House",
  },
  {
    id: 5,
    address: "202 Birch Rd, Anytown USA",
    beds: 5,
    baths: 4,
    area: 3000,
    style: "Bohemian",
    image: "https://via.placeholder.com/300x200?text=Bohemian+House",
  },
];

const FindHousePage = () => {
  const [beds, setBeds] = useState("");
  const [minArea, setMinArea] = useState("");
  const [maxArea, setMaxArea] = useState("");
  const [style, setStyle] = useState("any");
  const [searchResults, setSearchResults] = useState([]);

  const handleSearch = () => {
    const filteredResults = mockHouses.filter((house) => {
      const bedMatch = beds ? house.beds >= parseInt(beds) : true;
      const minAreaMatch = minArea ? house.area >= parseInt(minArea) : true;
      const maxAreaMatch = maxArea ? house.area <= parseInt(maxArea) : true;
      const styleMatch = style === "any" ? true : house.style === style;
      return bedMatch && minAreaMatch && maxAreaMatch && styleMatch;
    });
    setSearchResults(filteredResults);
  };

  return (
    <div className="min-h-screen bg-background py-8">
      <div className="container mx-auto px-4">
        <div className="text-left mb-8 pt-24 md:pt-28">
          <h1 className="text-4xl md:text-5xl font-bold text-text-primary mb-6 leading-tight">
            Find Your <span className="text-primary">Home</span>
          </h1>
          <p className="text-lg text-text-secondary mb-8">
            Search for houses that match your specific needs and interior design
            preferences.
          </p>

          <div className="bg-surface rounded-2xl shadow-2xl p-8 border border-border mb-12 relative overflow-hidden">
            <h2 className="text-2xl font-semibold text-text-primary mb-6">
              Search Filters
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <div>
                <label
                  htmlFor="beds"
                  className="block text-text-secondary text-sm font-bold mb-2"
                >
                  Min. Bedrooms:
                </label>
                <input
                  type="number"
                  id="beds"
                  value={beds}
                  onChange={(e) => setBeds(e.target.value)}
                  className="w-full p-3 bg-surface border border-border rounded-lg text-text-primary placeholder-text-secondary focus:ring-primary focus:border-primary"
                  placeholder="e.g., 3"
                />
              </div>
              <div>
                <label
                  htmlFor="minArea"
                  className="block text-text-secondary text-sm font-bold mb-2"
                >
                  Min. Area (sqft):
                </label>
                <input
                  type="number"
                  id="minArea"
                  value={minArea}
                  onChange={(e) => setMinArea(e.target.value)}
                  className="w-full p-3 bg-surface border border-border rounded-lg text-text-primary placeholder-text-secondary focus:ring-primary focus:border-primary"
                  placeholder="e.g., 1000"
                />
              </div>
              <div>
                <label
                  htmlFor="maxArea"
                  className="block text-text-secondary text-sm font-bold mb-2"
                >
                  Max. Area (sqft):
                </label>
                <input
                  type="number"
                  id="maxArea"
                  value={maxArea}
                  onChange={(e) => setMaxArea(e.target.value)}
                  className="w-full p-3 bg-surface border border-border rounded-lg text-text-primary placeholder-text-secondary focus:ring-primary focus:border-primary"
                  placeholder="e.g., 2500"
                />
              </div>
              <div>
                <label
                  htmlFor="style"
                  className="block text-text-secondary text-sm font-bold mb-2"
                >
                  Preferred Style:
                </label>
                <select
                  id="style"
                  value={style}
                  onChange={(e) => setStyle(e.target.value)}
                  className="w-full p-3 bg-surface border border-border rounded-lg text-text-primary placeholder-text-secondary focus:ring-primary focus:border-primary"
                >
                  <option value="any">Any</option>
                  <option value="Modern">Modern</option>
                  <option value="Minimalist">Minimalist</option>
                  <option value="Vintage">Vintage</option>
                  <option value="Industrial">Industrial</option>
                  <option value="Bohemian">Bohemian</option>
                </select>
              </div>
            </div>
            <button
              onClick={handleSearch}
              className="w-full bg-gradient-to-r from-primary to-secondary text-white px-8 py-4 rounded-xl font-semibold hover:shadow-lg hover:shadow-primary/30 transition-all duration-300 transform hover:-translate-y-1"
            >
              Search Houses
            </button>
          </div>

          <div className="mt-8">
            <h2 className="text-2xl font-semibold text-text-primary mb-4">
              Search Results
            </h2>
            {searchResults.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {searchResults.map((house) => (
                  <div
                    key={house.id}
                    className="group bg-surface rounded-2xl shadow-2xl overflow-hidden border border-border hover:border-primary/50 transition-all duration-500 hover:scale-105 hover:shadow-primary/20"
                  >
                    <img
                      src={house.image}
                      alt={house.address}
                      className="w-full h-48 object-cover group-hover:scale-110 transition-transform duration-500"
                    />
                    <div className="p-6">
                      <h3 className="text-xl font-bold text-text-primary mb-3 group-hover:text-primary transition-colors duration-300">
                        {house.address}
                      </h3>
                      <p className="text-text-secondary text-sm mb-2 leading-relaxed">
                        {house.beds} Beds | {house.baths} Baths | {house.area}{" "}
                        sqft
                      </p>
                      <p className="text-text-secondary text-sm">
                        Style: {house.style}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-text-secondary">
                No houses found. Adjust your search filters.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default FindHousePage;
