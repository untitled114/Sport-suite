{% macro map_book_name(book_id_column) %}
    case {{ book_id_column }}
        when 10 then 'FanDuel'
        when 12 then 'DraftKings'
        when 13 then 'Caesars'
        when 18 then 'BetRivers'
        when 19 then 'BetMGM'
        when 33 then 'ESPNBet'
        when 36 then 'Underdog'
        else 'Unknown'
    end
{% endmacro %}
