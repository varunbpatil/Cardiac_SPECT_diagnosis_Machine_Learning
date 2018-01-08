function movieList = loadMovieList()

%   movieList = loadMovieList() reads the fixed movie list in movie.txt 
%   and returns a cell array of the words in movieList.


%% open the movie_ids.txt file 
fid = fopen('movie_ids.txt');

% Store all movie names in cell array movieList{}
n = 1682;  % Total number of movies in the database

movieList = cell(n, 1);
for i = 1:n
    % Read a single line
    line = fgets(fid);
    % tokenize the line read
    [idx, movieName] = strtok(line, ' ');
    % Store only the movie name in the cell array movieList{}
    movieList{i} = strtrim(movieName);
end
fclose(fid);

end
